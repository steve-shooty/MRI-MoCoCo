import matplotlib.pyplot as plt

from ipywidgets import interact, IntSlider
import numpy as np
import SimpleITK as sitk
import cv2

def explore_3D_array(arr: np.ndarray, cmap: str = 'gray'):
  """
  Given a 3D array with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D array. 
  The purpose of this function to visual inspect the 2D arrays in the image. 

  Args:
    arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
    cmap : Which color map use to plot the slices in matplotlib.pyplot
  """

  def fn(SLICE):
    plt.figure(figsize=(7,7))
    plt.imshow(arr[SLICE, :, :], cmap=cmap)
    plt.show()

  interact(fn, SLICE=(0, arr.shape[0]-1))


def explore_3D_array_comparison(arr_before: np.ndarray, arr_after: np.ndarray, cmap: str = 'gray'):
  """
  Given two 3D arrays with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D arrays.
  The purpose of this function to visual compare the 2D arrays after some transformation. 

  Args:
    arr_before : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, before any transform
    arr_after : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, after some transform    
    cmap : Which color map use to plot the slices in matplotlib.pyplot
  """

  assert arr_after.shape == arr_before.shape

  def fn(SLICE):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10,10))

    ax1.set_title('Before', fontsize=15)
    ax1.imshow(arr_before[SLICE, :, :], cmap=cmap)

    ax2.set_title('After', fontsize=15)
    ax2.imshow(arr_after[SLICE, :, :], cmap=cmap)

    plt.tight_layout()
    plt.show()
  
  interact(fn, SLICE=(0, arr_before.shape[0]-1))


def explore_3D_array_comparison_and_diff(arr_before: np.ndarray, arr_after: np.ndarray, cmap: str = 'gray'):
    """
    Given two 3D arrays with shape (Z,X,Y) this function will create an interactive
    widget to check out all the 2D arrays with shape (X,Y) inside the 3D arrays.
    The purpose of this function to visual compare the 2D arrays and their difference.

    Args:
      arr_before : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, before any transform
      arr_after : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, after some transform
      cmap : Which color map use to plot the slices in matplotlib.pyplot
    """
    assert arr_after.shape == arr_before.shape, "Die beiden Arrays müssen die gleiche Form haben."

    # Berechne das Differenz-Array einmal im Voraus für bessere Performance
    diff_arr = np.absolute(arr_before - arr_after)

    # Die Funktion, die bei jeder Slider-Änderung aufgerufen wird
    def fn(SLICE):
        # Erstelle eine Figur mit 1 Zeile und 3 Spalten für die Plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(15, 7))

        # Plot 1: Bild "Vorher"
        ax1.set_title('Before', fontsize=15)
        im1 = ax1.imshow(arr_before[SLICE, :, :], cmap=cmap)
        ax1.axis('off')

        # Plot 2: Bild "Nachher"
        ax2.set_title('After', fontsize=15)
        im2 = ax2.imshow(arr_after[SLICE, :, :], cmap=cmap)
        ax2.axis('off')
        
        # Plot 3: Differenzbild
        ax3.set_title('Difference (Absolute)', fontsize=15)
        # Wir verwenden eine andere Colormap ('magma' oder 'hot'), um Unterschiede hervorzuheben.
        # Schwarz = kein Unterschied, Helle Farben = großer Unterschied.
        im3 = ax3.imshow(diff_arr[SLICE, :, :], cmap='magma')
        ax3.axis('off')

        plt.tight_layout()
        plt.show()
    
    # Erstelle den interaktiven Slider
    interact(fn, SLICE=IntSlider(min=0, max=arr_before.shape[0]-1, step=1, value=arr_before.shape[0]//2, description='Slice:'))

def show_sitk_img_info(img: sitk.Image):
  """
  Given a sitk.Image instance prints the information about the MRI image contained.

  Args:
    img : instance of the sitk.Image to check out
  """
  pixel_type = img.GetPixelIDTypeAsString()
  origin = img.GetOrigin()
  dimensions = img.GetSize()
  spacing = img.GetSpacing()
  direction = img.GetDirection()

  info = {'Pixel Type' : pixel_type, 'Dimensions': dimensions, 'Spacing': spacing, 'Origin': origin,  'Direction' : direction}
  for k,v in info.items():
    print(f' {k} : {v}')


def add_suffix_to_filename(filename: str, suffix:str) -> str:
  """
  Takes a NIfTI filename and appends a suffix.

  Args:
      filename : NIfTI filename
      suffix : suffix to append

  Returns:
      str : filename after append the suffix
  """
  if filename.endswith('.nii'):
      result = filename.replace('.nii', f'_{suffix}.nii')
      return result
  elif filename.endswith('.nii.gz'):
      result = filename.replace('.nii.gz', f'_{suffix}.nii.gz')
      return result
  else:
      raise RuntimeError('filename with unknown extension')


def rescale_linear(array: np.ndarray, new_min: int, new_max: int):
  """Rescale an array linearly."""
  minimum, maximum = np.min(array), np.max(array)
  m = (new_max - new_min) / (maximum - minimum)
  b = new_min - m * minimum
  return m * array + b


def explore_3D_array_with_mask_contour(arr: np.ndarray, mask: np.ndarray, thickness: int = 1):
  """
  Given a 3D array with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D array. The binary
  mask provided will be used to overlay contours of the region of interest over the 
  array. The purpose of this function is to visual inspect the region delimited by the mask.

  Args:
    arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
    mask : binary mask to obtain the region of interest
  """
  assert arr.shape == mask.shape
  
  _arr = rescale_linear(arr,0,1)
  _mask = rescale_linear(mask,0,1)
  _mask = _mask.astype(np.uint8)

  def fn(SLICE):
    arr_rgb = cv2.cvtColor(_arr[SLICE, :, :], cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(_mask[SLICE, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    arr_with_contours = cv2.drawContours(arr_rgb, contours, -1, (0,1,0), thickness)

    plt.figure(figsize=(7,7))
    plt.imshow(arr_with_contours)
    plt.show()

  interact(fn, SLICE=(0, arr.shape[0]-1))
  
def explore_orthogonal_views(arr: np.ndarray, metadata: dict, cmap: str = 'gray'):
    """
    Given a 3D array (Z, Y, X), this function creates an interactive widget 
    with sliders to explore the orthogonal views (axial, coronal, sagittal).

    Args:
        arr (np.ndarray): 3D array with shape (Z, Y, X) representing the MRI volume.
        metadata (dict): Dictionary containing 'PixelSpacing' and 'SliceThickness'.
        cmap (str): Colormap for matplotlib.pyplot.
    """
    
    # Get the dimensions of the array
    max_z, max_y, max_x = arr.shape[0] - 1, arr.shape[1] - 1, arr.shape[2] - 1
    
    # Define the plotting function that interact will call
    def fn(z_slice, y_slice, x_slice):
        # Create three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # --- 1. Axial View ---
        axial_img = arr[z_slice, :, :]
        # Use metadata for correct aspect ratio
        aspect_axial = metadata['PixelSpacing'][1] / metadata['PixelSpacing'][0]
        axes[0].imshow(axial_img, cmap=cmap, aspect=aspect_axial)
        axes[0].set_title(f'Axial View (Slice Z = {z_slice})')
        # Add lines indicating the position of the other slices
        axes[0].axvline(x=x_slice, color='red', linewidth=0.8)
        axes[0].axhline(y=y_slice, color='lime', linewidth=0.8)

        # --- 2. Coronal View ---
        coronal_img = arr[:, y_slice, :]
        aspect_coronal = metadata['SliceThickness'] / metadata['PixelSpacing'][0]
        axes[1].imshow(coronal_img, cmap=cmap, aspect=aspect_coronal)
        axes[1].set_title(f'Coronal View (Slice Y = {y_slice})')
        axes[1].axvline(x=x_slice, color='red', linewidth=0.8)
        axes[1].axhline(y=z_slice, color='cyan', linewidth=0.8)

        # --- 3. Sagittal View ---
        sagittal_img = arr[:, :, x_slice]
        aspect_sagittal = metadata['SliceThickness'] / metadata['PixelSpacing'][1]
        # Use .T and origin to match orientation from your original code
        axes[2].imshow(sagittal_img.T, cmap=cmap, origin='lower', aspect=aspect_sagittal)
        axes[2].set_title(f'Sagittal View (Slice X = {x_slice})')
        axes[2].axhline(y=y_slice, color='lime', linewidth=0.8)
        axes[2].axvline(x=z_slice, color='cyan', linewidth=0.8)
        
        plt.tight_layout()
        plt.show()

    # Create the interactive widget with three sliders
    interact(
        fn, 
        z_slice=IntSlider(min=0, max=max_z, step=1, value=max_z // 2, description='Axial (Z):'),
        y_slice=IntSlider(min=0, max=max_y, step=1, value=max_y // 2, description='Coronal (Y):'),
        x_slice=IntSlider(min=0, max=max_x, step=1, value=max_x // 2, description='Sagittal (X):')
    )