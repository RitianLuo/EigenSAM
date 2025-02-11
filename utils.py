import torch
from torch.autograd.functional import hvp
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def normalization(tensors):
    """
    Normalize a tuple of tensors so that their combined norm is 1.

    Args:
        tensors (Tuple[torch.Tensor, ...]): Tensors of arbitrary shapes.

    Returns:
        Tuple[torch.Tensor, ...]: A new tuple of tensors with the same shapes,
            scaled so that their total L2 norm becomes 1.
    """
    # Compute the sum of squares across all tensors
    sq_sum = sum((t**2).sum() for t in tensors)
    norm = torch.sqrt(sq_sum + 1e-12)
    # Divide each tensor by the norm
    return tuple(t / norm for t in tensors)

def estimate_largest_eigenvector(model, criterion, v, images, labels, steps=5):
    """
    Estimate the largest eigenvector of the Hessian using the Power Method.

    Args:
        model (torch.nn.Module): The model.
        criterion (callable): Loss function.
        v (list[Tensor] or tuple[Tensor]): Initial vector (same shape as model parameters).
        images (torch.Tensor): Input batch.
        labels (torch.Tensor): Corresponding labels.
        steps (int): Power Method iterations.

    Returns:
        list[torch.Tensor]: Approximate dominant eigenvector of the Hessian.
    """
    if v is None:
        v = tuple(torch.randn_like(p) for p in model.parameters() if p.requires_grad)

    model.eval()  # Disable dropout, batch norm updates
    loss = criterion(model(images), labels)  # Compute loss
    loss.backward(create_graph=True)  # First-order gradients

    params, gradsH = get_params_grad(model)  # Get parameters and gradients

    for i in range(steps):
        hvp = torch.autograd.grad(gradsH, params, grad_outputs=v, retain_graph=(i < steps - 1))
        v = normalization(hvp)  # Normalize for next iteration

    model.zero_grad(set_to_none=True)  # Clear gradients
    model.train()  # Restore train mode

    return v

def modify_gradient_with_projection(model, v, alpha=0.1):
    """
    Perform a "global projection" in the gradient space. This function manually flattens
    the gradients, applies the projection logic, and unflattens them back:

    Steps:
      1) Flatten all non-None gradients into g_flat.
      2) Flatten 'v' into v_flat in the same order.
      3) Compute the global dot product (g_flat dot v_flat), then its sign.
      4) Normalize g_flat to have unit length.
      5) Compute the vertical component of v_flat w.r.t. g_flat.
      6) Add alpha * sign_gv * that vertical component to g_flat.
      7) Reshape g_flat back into each parameter's .grad.

    Args:
        model (torch.nn.Module): The model, which must have .grad for each parameter 
            you want to modify.
        v (Tuple[torch.Tensor, ...] or list[torch.Tensor]): A vector matching
            model.parameters() in shape/order.
        alpha (float): Scaling factor for how much of the vertical component is added.
    """
    with torch.no_grad():
        # 1) Gather gradients into a list and flatten
        params_with_grad = []
        grads_list = []
        for p in model.parameters():
            if p.grad is not None:
                params_with_grad.append(p)
                grads_list.append(p.grad.view(-1))

        if len(grads_list) == 0:
            return  # No gradients to modify

        g_flat = torch.cat(grads_list, dim=0)  # shape: (total_grad_size,)

        # 2) Flatten v in the same order
        v_list = []
        for idx, p in enumerate(params_with_grad):
            # Each v_i must match p's shape
            v_i = v[idx]
            v_list.append(v_i.reshape(-1))
        v_flat = torch.cat(v_list, dim=0)  # shape: (total_grad_size,)

        # 3) Global dot product & sign
        dot = g_flat.dot(v_flat)
        sign_gv = torch.sign(dot)

        # 4) Normalize g_flat to length 1
        g_norm = g_flat.norm() + 1e-12
        g_flat /= g_norm

        # 5) Compute vertical component of v_flat w.r.t. g_flat
        dot_normed = g_flat.dot(v_flat)
        v_vertical = v_flat - dot_normed * g_flat

        # 6) Add alpha * sign_gv * v_vertical
        g_flat += alpha * sign_gv * v_vertical

        # 7) Unflatten g_flat back to each parameter's .grad
        pointer = 0
        for p, original_grad in zip(params_with_grad, grads_list):
            numel = original_grad.numel()
            new_slice = g_flat[pointer : pointer + numel]
            pointer += numel

            p.grad.data.copy_(new_slice.view_as(p.grad))

def get_params_grad(model):
    """
    Collects all parameters from the model that require a gradient, along with
    their gradient tensors. If a parameter's .grad is None, it creates a zero
    tensor of the same shape as a placeholder.

    Returns:
        params (list[Tensor]): The list of parameters with requires_grad=True.
        grads  (list[Tensor]): The list of corresponding gradient tensors.
            - If .grad is not None, a detached clone of the gradient is stored.
            - If .grad is None, a zero tensor (same shape) is used.
    """
    # Filter out parameters that do not require gradient
    params = [p for p in model.parameters() if p.requires_grad]

    # For each parameter, either clone its gradient or create a zero tensor
    grads = [
        p.grad.clone() if p.grad is not None
        else torch.zeros_like(p)
        for p in params
    ]

    return params, grads
