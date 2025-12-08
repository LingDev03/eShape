import torch.nn as nn
import torch
import numpy
import torch.nn.functional as F

"""
------------------------------------------------------------------------------------------------------------------------
Original Learning TS Shapelets
------------------------------------------------------------------------------------------------------------------------
"""
class EuclidDistBlock(nn.Module):
    def __init__(self, shapelet, shapelet_info=None, len_ts=5, alpha=-10):
        super(PISDistBlock, self).__init__()
        self.alpha = alpha
        self.len_ts = len_ts
        sc = torch.FloatTensor(shapelet)
        self.shapelet = nn.Parameter(sc.view(1,sc.size(-1)))
        self.kernel_size = self.shapelet.size(-1)
        self.out_channels = len_ts - self.shapelet.size(-1) + 1
        self.false_conv_layer = nn.Conv1d(in_channels=2,
                                          out_channels=2,
                                          kernel_size=self.kernel_size,
                                          bias=False)
        data = torch.Tensor(numpy.eye(self.kernel_size))
        self.false_conv_layer.weight.data = data.view(self.kernel_size,
                                                      1,
                                                      self.kernel_size)
        for p in self.false_conv_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        reshaped_x1 = self.false_conv_layer(x)
        reshaped_x2 = torch.transpose(reshaped_x1, 1, 2)
        reshaped_x3 = reshaped_x2.contiguous().view(-1, self.kernel_size)
        dist1 = torch.sum(torch.square(reshaped_x3 - self.shapelet),1)/self.shapelet.size(-1)
        dist2 = dist1.view(x.size(0), 1, self.out_channels)

        # soft-minimum
        dist3 = self.soft_minimum(dist2)

        return dist3

    def soft_minimum(self, dist):
        temp = torch.exp(self.alpha * dist)
        return torch.sum(temp*dist, 2)/torch.sum(temp, 2)

    def hard_minimum(self, dist):
        min_dist, _ = torch.min(dist, 2)
        return min_dist

    def get_shapelets(self):
        return self.shapelet


class ShapeletLayer(nn.Module):
    def __init__(self, shapelets , len_ts):
        super(ShapeletLayer, self).__init__()
        self.blocks = nn.ModuleList([EuclidDistBlock(shapelet=shapelets[i],len_ts=len_ts)
                                     for i in range(len(shapelets))])

    def forward(self, x):
        out = torch.FloatTensor([]).to(x.device)
        for block in self.blocks:
            out = torch.cat((out, block(x)), dim=1)

        return out.view(out.size(0),1,out.size(1))

"""
------------------------------------------------------------------------------------------------------------------------
Perceptually and Position-aware Learning TS Shapelets
------------------------------------------------------------------------------------------------------------------------
"""
class PISDistBlock(nn.Module):
    """
    Parameter:
    shaplet:
    shaplet_info:
    in_chanels: input
    """
    def __init__(self, shapelet, shapelet_info=None, len_ts=5, alpha=-10,
                 window_size=10, norm=10, bounding_norm=100, maximum_ci=3):
        super(PISDistBlock, self).__init__()
        self.alpha = alpha
        self.norm = norm
        self.len_ts = len_ts
        
        # Use window_size from shapelet_info if available (index 5), otherwise use the parameter
        if shapelet_info is not None and len(shapelet_info) > 5:
            self.window_size = int(shapelet_info[5])
        else:
            self.window_size = window_size
            
        self.bounding_norm = bounding_norm
        self.max_norm_dist = nn.Parameter(torch.tensor(0.00001), requires_grad=False)
        self.maximum_ci = maximum_ci

        self.start_position = int(shapelet_info[1] - self.window_size)
        self.start_position = self.start_position if self.start_position >= 0 else 0
        self.end_position = int(shapelet_info[2] + self.window_size)
        self.end_position = self.end_position if self.end_position < len_ts else len_ts

        sc = torch.FloatTensor(shapelet)
        self.shapelet = nn.Parameter(sc.view(1,sc.size(-1)), requires_grad=True)
        self.kernel_size = self.shapelet.size(-1)
        
        # Calculate out_channels and ensure it's at least 1
        self.out_channels = self.end_position - self.start_position - self.shapelet.size(-1) + 1
        
        # If out_channels is <= 0, adjust the window to ensure valid computation
        if self.out_channels <= 0:
            # Expand the window to ensure at least 1 output channel
            required_window = self.shapelet.size(-1)
            self.start_position = max(0, int(shapelet_info[1]) - required_window // 2)
            self.end_position = min(len_ts, self.start_position + required_window)
            # Recalculate out_channels
            self.out_channels = self.end_position - self.start_position - self.shapelet.size(-1) + 1
            # Ensure it's at least 1
            if self.out_channels <= 0:
                self.end_position = self.start_position + self.shapelet.size(-1)
                self.out_channels = 1
        # No need for Conv1d layers - we'll use unfold instead
        # This is more efficient and avoids cuDNN issues

    def forward(self, x, ep):
        try:
            self.ci_shapelet = torch.sum(torch.square(torch.subtract(self.shapelet.data.detach()[:,1:],
                                                                     self.shapelet.data.detach()[:,:-1]))) + (1/self.norm)

            pis = x[:,:,self.start_position:self.end_position]
            
            # Safety check: ensure pis has enough length for shapelet comparison
            if pis.size(-1) < self.kernel_size:
                # If the position-aware segment is smaller than shapelet, use full time series
                pis = x
            
            # Additional safety: ensure pis has at least kernel_size length
            if pis.size(-1) < self.kernel_size:
                # Return a default output if still too small
                return torch.zeros(x.size(0), 1, 1).to(x.device)
                
            ci_pis = torch.square(torch.subtract(pis[:,:,1:], pis[:,:,:-1]))

            # Check if we can extract sliding windows
            expected_output_size = pis.size(-1) - self.kernel_size + 1
            if expected_output_size <= 0:
                return torch.zeros(x.size(0), 1, 1).to(x.device)

            # Use unfold to extract sliding windows instead of Conv1d
            # unfold(dimension, size, step) extracts sliding windows
            # Input: (batch, 1, length) -> Output: (batch, 1, num_windows, kernel_size)
            reshaped_pis1 = pis.unfold(2, self.kernel_size, 1)  # (batch, 1, num_windows, kernel_size)
            reshaped_pis1 = reshaped_pis1.squeeze(1)  # (batch, num_windows, kernel_size)
            reshaped_pis1 = reshaped_pis1.contiguous().view(-1, self.kernel_size)  # (batch*num_windows, kernel_size)

            if self.kernel_size > 1:
                # Same for CI
                reshaped_ci_pis1 = ci_pis.unfold(2, self.kernel_size - 1, 1)
                reshaped_ci_pis1 = reshaped_ci_pis1.squeeze(1)
                reshaped_ci_pis1 = reshaped_ci_pis1.contiguous().view(-1, self.kernel_size - 1)
                reshaped_ci_pis1 = torch.sum(reshaped_ci_pis1, dim=1) + (1/self.norm)
            else:
                # For kernel_size=1, ci calculation is trivial
                reshaped_ci_pis1 = torch.ones(reshaped_pis1.size(0)).to(x.device) * (1/self.norm)

            ci_shapelet_vec = self.ci_shapelet.repeat(reshaped_ci_pis1.size(0))
            max_ci = torch.max(reshaped_ci_pis1, ci_shapelet_vec)
            min_ci = torch.min(reshaped_ci_pis1, ci_shapelet_vec)
            
            # Avoid division by zero in ci_dist calculation
            min_ci = torch.clamp(min_ci, min=1e-8)
            ci_dist = max_ci / min_ci
            ci_dist = torch.clamp(ci_dist, max=self.maximum_ci)
            
            # Replace NaN and Inf values
            ci_dist = torch.nan_to_num(ci_dist, nan=1.0, posinf=self.maximum_ci, neginf=1.0)
            
            dist1 = torch.sum(torch.square(reshaped_pis1 - self.shapelet),1)
            dist1 = dist1 * ci_dist
            dist1 = dist1 / self.shapelet.size(-1)
            
            # Replace NaN and Inf values in dist1
            dist1 = torch.nan_to_num(dist1, nan=0.0, posinf=1e10, neginf=0.0)
            
            # Recalculate actual out_channels based on actual pis size
            actual_out_channels = pis.size(-1) - self.kernel_size + 1
            if actual_out_channels <= 0:
                return torch.zeros(x.size(0), 1, 1).to(x.device)
            
            dist1 = dist1.view(x.size(0), 1, actual_out_channels)

            # soft-minimum with safety
            dist1 = self.soft_minimum(dist1)
            
            # Replace NaN and Inf after soft_minimum
            dist1 = torch.nan_to_num(dist1, nan=0.0, posinf=1e10, neginf=0.0)

            if ep == 0 and self.training:
                # Ensure dist1 is valid before computing max
                if dist1.numel() > 0 and torch.isfinite(dist1).all():
                    max_value = torch.max(dist1.detach()).item()
                    max_norm_value = self.max_norm_dist.item()
                    if max_value > max_norm_value and max_value < 1e10:
                        self.max_norm_dist.data = torch.tensor(max_value).to(self.max_norm_dist.device)
            
            # Safety check: ensure max_norm_dist is not zero to avoid division by zero
            max_norm_value = self.max_norm_dist.item()
            if max_norm_value > 1e-8:
                dist1 = 1 - dist1/self.max_norm_dist
                dist1 = torch.clamp(dist1, min=0.0, max=1.0)  # Ensure output is in [0, 1]
            else:
                dist1 = torch.zeros_like(dist1)

            return dist1
            
        except RuntimeError as e:
            # If any error occurs, return zero tensor
            print(f"Warning: Error in PISDistBlock.forward: {e}")
            return torch.zeros(x.size(0), 1, 1).to(x.device)

    def soft_minimum(self, dist):
        try:
            dist1 = dist / self.bounding_norm
            # Clamp to avoid exp overflow
            dist1 = torch.clamp(dist1, min=-50, max=50)
            temp = torch.exp(self.alpha * dist1)
            
            # Avoid division by zero
            sum_temp = torch.sum(temp, 2)
            sum_temp = torch.clamp(sum_temp, min=1e-8)
            
            min_dist = torch.sum(temp*dist1, 2) / sum_temp
            min_dist = min_dist * self.bounding_norm
            
            # Replace NaN and Inf values
            min_dist = torch.nan_to_num(min_dist, nan=0.0, posinf=self.bounding_norm, neginf=0.0)
            
            return min_dist
        except RuntimeError:
            # Fallback to hard minimum if soft minimum fails
            return self.hard_minimum(dist)

    def hard_minimum(self, dist):
        min_dist, _ = torch.min(dist, 2)
        return min_dist

    def get_shapelets(self):
        return self.shapelet


class PShapeletLayer(nn.Module):
    def __init__(self, shapelets_info, shapelets , len_ts, window_size=20, bounding_norm=100):
        super(PShapeletLayer, self).__init__()
        self.blocks = nn.ModuleList([
            PISDistBlock(shapelet=shapelets[i],shapelet_info=shapelets_info[i],len_ts=len_ts,window_size=window_size,
                         bounding_norm=bounding_norm)
            for i in range(len(shapelets))])

    def transform_to_complexity_invariance(self, x):
        return torch.square(torch.subtract(x[:, :, 1:], x[:, :, :-1]))

    def forward(self, x, ep):
        out = torch.FloatTensor([]).to(x.device)
        for block in self.blocks:
            out = torch.cat((out, block(x,ep=ep)), dim=1)

        return out.view(out.size(0),1,out.size(1))


class LearningPShapeletsModel(nn.Module):
    def __init__(self, shapelets_info, shapelets , len_ts, num_classes, sge=0, window_size=20, bounding_norm=100):
        super(LearningPShapeletsModel, self).__init__()
        self.sge = sge
        self.pshapelet_layer = PShapeletLayer(shapelets_info=shapelets_info, shapelets=shapelets,len_ts=len_ts,
                                              window_size=window_size,bounding_norm=bounding_norm)
        self.num_shapelets = len(shapelets)
        self.linear3 = nn.Linear(self.num_shapelets, num_classes)

    def forward(self, x, ep):
        y = self.pshapelet_layer(x,ep)
        y = torch.relu(y)
        if ep < self.sge:
            y = self.linear3(y.detach())
        else:
            y = self.linear3(y)
        y = torch.squeeze(y, 1)
        return y


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # time_series = torch.Tensor([[[1., 2., 3., 4., 5.],[1., 1., 1., 1., 1.]], [[-4., -5., -6., -7., -8.],[-2., -2., -2., -2., -2.]]]).view(2,2,5)
    time_series = torch.Tensor([[1., 2., 3., 4., 5., 6., 7., 4., 5.], [-2., -2., -2., -2., -3., 4., 5., 4., 5.]]).view(2,1,9)
    shapelets = [[1., 2., 3.], [3., 4., 5.], [5., 6., 6.]]
    shapelets_info = numpy.array([[1., 1., 4., 4., 5.], [1., 1., 3., 4., 5.], [1., 2., 3., 4., 5.]])
    len_ts = time_series.size(-1)

    layer = PShapeletLayer(shapelets_info=shapelets_info, shapelets=shapelets,len_ts=len_ts, window_size=1).to("cuda:0")
    time_series = time_series.to(device)
    dists = layer.forward(time_series, ep=1)
    print(dists)