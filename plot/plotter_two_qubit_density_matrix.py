import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_density_matrix_bars(matrix, title, cmap='bwr', 
                             vmin=-1, vmax=1, ax=None, labels=None,
                             annotate=True):
    """
    Plots a 3D bar chart of a matrix component (real, imaginary, or magnitude).

    Parameters:
        matrix (np.ndarray): 2D array representing the matrix component to plot.
        title (str): Title of the subplot.
        cmap (str): Colormap to use for the bars.
        vmin (float): Minimum value for color normalization.
        vmax (float): Maximum value for color normalization.
        ax (matplotlib.axes._subplots.Axes3DSubplot): 3D subplot axis to draw on.
        labels (list): List of basis labels for x and y axes.
        annotate: bool, add value text at each bar
    """
    n = matrix.shape[0]
    assert matrix.shape[1] == n, "Matrix must be square"
    
    # Prepare 3D axis
    if ax is None:
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
    
    # Use 'xy' indexing for orientation consistency
    xpos, ypos = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    xpos = xpos.ravel(order='C')
    ypos = ypos.ravel(order='C')
    dz   = matrix.ravel(order='C')
    
    # Normalize colors over the fixed range
    norm     = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    colors = mappable.to_rgba(dz)
    colors[:, -1] = 0.8  # semi-transparent bars
    
    # Bars extend up or down from zero
    bottoms = np.where(dz < 0, dz, 0)
    heights = np.abs(dz)

    # Add a gray plane at z=0
    xx, yy = np.meshgrid(np.linspace(-0.5, n+0.3, 10),
                         np.linspace(-0.5, n+0.3, 10), indexing='xy')
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='gray', alpha=0.25, linewidth=0)
    
    # Draw bars
    ax.bar3d(xpos, ypos, bottoms,
             0.7, 0.7, heights,
             color=colors,
             edgecolor='k',  # black border
             linewidth=0.5,
             shade=True)
    
    # Ticks & labels
    ax.set_xticks(np.arange(n) + 0.35)
    ax.set_yticks(np.arange(n) + 0.35)
    if labels is not None:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    #ax.set_zlabel('Value')
    ax.set_title(title)

    # Fix z-axis limits
    ax.set_zlim(vmin, vmax)
    ax.view_init(elev=25, azim=-60)

    # Colorbar legend
    plt.colorbar(mappable, ax=ax, fraction=0.03,
                 pad=0.1, label='Value')
    
    # Annotate values
    if annotate:
        for x, y, v in zip(xpos, ypos, dz):
            txt = f"{v:.3f}"
            z = v if v >= 0 else v - 0.04 * (vmax - vmin)
            ax.text(x + 0.35, y + 0.35, z + (0.02 if v >= 0 else -0.01),
                    txt, ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=8, color='black', rotation=0)

def plot_density_matrix(matrix, 
                        state_label='Density Matrix', 
                        basis_labels=None, 
                        components=['real', 'imag', 'abs'],
                        zmin=-1.0, zmax=1.0,
                        annotate=False):
    """
    Plot the selected components of a 4×4 density matrix in 3D bar charts,
    using a fixed z-axis and color range [zmin, zmax].

    Parameters:
        matrix       : 4×4 complex density matrix.
        state_label  : super-title for the figure.
        basis_labels : list of 4 strings for x/y axes 
                        (default ['HH','HV','VH','VV']).
        components   : subset of ['real','imag','abs'] to plot.
        zmin, zmax   : fixed limits for z axis and colormap.
        annotate     : bool, add value text at each bar.
    """
    # Default basis
    if basis_labels is None:
        basis_labels = ['HH','HV','VH','VV']

    # Map component → (transform, subplot title, colormap)
    cmap_map = {
        'real': (np.real, 'Real Part',       'managua'),
        'imag': (np.imag, 'Imaginary Part', 'managua'),
        'abs' : (np.abs,  'Magnitude',      'managua')
    }

    chosen = [c for c in components if c in cmap_map]
    if not chosen:
        raise ValueError("No valid components selected!")

    fig = plt.figure(figsize=(6*len(chosen), 5))
    fig.suptitle(state_label, y=1.02)

    # Plot each component in its own subplot
    for i, comp in enumerate(chosen):
        func, title, cmap = cmap_map[comp]
        ax = fig.add_subplot(1, len(chosen), i+1, projection='3d')
        data = func(matrix)
        plot_density_matrix_bars(data, title, cmap=cmap,
                                 vmin=zmin, vmax=zmax,
                                 ax=ax, labels=basis_labels,
                                 annotate=annotate)

    #plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example: mixed two-qubit state = 50% |ψ+⟩ + 50% |ψ-⟩
    psi_p = np.zeros(4, complex)
    psi_m = np.zeros(4, complex)
    psi_p[1] = psi_p[2] = 1/np.sqrt(2)
    psi_m[1] =  1/np.sqrt(2)
    psi_m[2] = -1/np.sqrt(2)

    rho_p = np.outer(psi_p, psi_p.conj())
    rho_m = np.outer(psi_m, psi_m.conj())
    rho_mix = 0.5*rho_p + 0.5*rho_m

    # Plot real and magnitude, with z axis fixed to [-1,1]
    plot_density_matrix(rho_mix,
                        state_label=r"Mixed: 50% $|\psi^+\rangle$ + 50% $|\psi^-\rangle$",
                        components=['real','abs'],
                        zmin=-1, zmax=1)

