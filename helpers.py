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

    diff_arr = np.absolute(arr_before - arr_after)

    def fn(SLICE):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(15, 7))

        ax1.set_title('Before', fontsize=15)
        im1 = ax1.imshow(arr_before[SLICE, :, :], cmap=cmap)
        ax1.axis('off')

        ax2.set_title('After', fontsize=15)
        im2 = ax2.imshow(arr_after[SLICE, :, :], cmap=cmap)
        ax2.axis('off')

        ax3.set_title('Difference (Absolute)', fontsize=15)

        im3 = ax3.imshow(diff_arr[SLICE, :, :], cmap='magma')
        ax3.axis('off')

        plt.tight_layout()
        plt.show()

    interact(fn, SLICE=IntSlider(min=0, max=arr_before.shape[0]-1, step=1, value=arr_before.shape[0]//2, description='Slice:'))

def explore_3D_array_comparison_with_diff(
    arr1: np.ndarray, 
    arr2: np.ndarray, 
    arr3: np.ndarray, 
    cmap: str = 'gray'
):
    """
    Erstellt ein interaktives Widget, um Slices aus drei 3D-Arrays zu vergleichen.
    Zeigt die drei Bilder nebeneinander an und berechnet zusätzlich die absolute 
    Differenz zwischen dem zweiten (mittleren) und dritten (rechten) Bild.

    Args:
      arr1: Erstes 3D-Array (links), z.B. das Originalbild.
      arr2: Zweites 3D-Array (mitte), z.B. nach einer ersten Transformation.
      arr3: Drittes 3D-Array (rechts), z.B. nach einer zweiten Transformation.
      cmap: Farbkarte für die Darstellung der Bilder.
    """
    assert arr1.shape == arr2.shape == arr3.shape, "Alle drei Arrays müssen exakt die gleiche Form haben."

    diff_arr = np.absolute(arr2 - arr3)

    def fn(SLICE):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(20, 7))

        ax1.set_title('Image 1 (Left)', fontsize=15)
        ax1.imshow(arr1[SLICE, :, :], cmap=cmap)
        ax1.axis('off')

        ax2.set_title('Image 2 (Middle)', fontsize=15)
        ax2.imshow(arr2[SLICE, :, :], cmap=cmap)
        ax2.axis('off')

        ax3.set_title('Image 3 (Right)', fontsize=15)
        ax3.imshow(arr3[SLICE, :, :], cmap=cmap)
        ax3.axis('off')

        ax4.set_title('Difference (|Middle - Right|)', fontsize=15)
        im4 = ax4.imshow(diff_arr[SLICE, :, :], cmap='magma') 
        ax4.axis('off')

        plt.tight_layout()
        plt.show()

    interact(fn, SLICE=IntSlider(min=0, max=arr1.shape[0]-1, step=1, value=arr1.shape[0]//2, description='Slice:'))

def plot_time_sequence_comparison(
    arr1: np.ndarray, 
    arr2: np.ndarray, 
    t_start: int, 
    num_frames: int = 20,
    *, 
    cmap: str = 'gray', 
    diff_cmap: str = 'inferno' 
):
    """
    Erstellt einen Plot, der Slices aus zwei 3D-Arrays (T, H, W) 
    über ein Zeitfenster vergleicht. Zeigt arr1, arr2 und deren Differenz 
    untereinander für die spezifizierten Zeitpunkte an.

    Args:
      arr1: Erstes 3D-Array (T, H, W), z.B. das Originalbild.
      arr2: Zweites 3D-Array (T, H, W), z.B. nach einer Transformation.
      t_start: Der Startzeitpunkt für das Fenster.
      num_frames: Maximale Anzahl der anzuzeigenden Zeitpunkte.
      cmap: Farbkarte für die Darstellung von arr1 und arr2.
      diff_cmap: Farbkarte für das Differenzbild.
    """
    assert arr1.shape == arr2.shape, "Die beiden Arrays müssen exakt die gleiche Form haben."
    assert arr1.ndim == 3, "Eingabearrays müssen 3-dimensional sein (T, H, W)."
    
    T, H, W = arr1.shape
    t_end = min(t_start + num_frames, T)
    actual_frames = t_end - t_start

    if actual_frames <= 0:
        print(f"FEHLER: Startzeitpunkt {t_start} liegt außerhalb der gültigen Zeitreihe (Länge {T}).")
        return

    fig, axes = plt.subplots(actual_frames, 3, figsize=(15, 3 * actual_frames), 
                             sharex=True, sharey=True)

    if actual_frames == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f"Vergleich von Zeitpunkten {t_start} bis {t_end-1}", fontsize=16, y=1.0)
    print(f"Zeige {actual_frames} Zeitpunkte von {t_start} bis {t_end-1} an...")

    max_diff_overall = 0
    diffs = []
    for i in range(actual_frames):
        t = t_start + i
        slice1 = arr1[t, :, :]
        slice2 = arr2[t, :, :]
        diff_slice = np.absolute(slice1.astype(float) - slice2.astype(float))
        diffs.append(diff_slice)
        current_max = diff_slice.max() if diff_slice.size > 0 else 0
        if current_max > max_diff_overall:
            max_diff_overall = current_max

    for i in range(actual_frames):
        t = t_start + i
        
        slice1 = arr1[t, :, :]
        slice2 = arr2[t, :, :]
        diff_slice = diffs[i]

        axes[i, 0].imshow(slice1, cmap=cmap)
        axes[i, 0].set_ylabel(f'Time {t}', fontsize=12) 

        axes[i, 1].imshow(slice2, cmap=cmap)

        im_diff = axes[i, 2].imshow(diff_slice, cmap=diff_cmap, vmin=0, vmax=max_diff_overall)

        if i == 0:
            axes[i, 0].set_title('Array 1', fontsize=14)
            axes[i, 1].set_title('Array 2', fontsize=14)
            axes[i, 2].set_title('|Array 1 - Array 2|', fontsize=14)

        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95)
    plt.show()


dummy_arr1 = np.zeros((T, H, W), dtype=np.uint8)
dummy_arr2 = np.zeros((T, H, W), dtype=np.uint8)


y, x = np.ogrid[-H//2:H//2, -W//2:W//2]
radius = 30
for t in range(T):
    offset_x = int(15 * np.sin(t * 2 * np.pi / T))
    offset_y = int(10 * np.cos(t * 2 * np.pi / T))
    mask1 = (x - offset_x)**2 + (y - offset_y)**2 <= radius**2 
    mask2 = (x - offset_x//2)**2 + (y - offset_y//2)**2 <= radius**2
    dummy_arr1[t, :, :][mask1] = 200 
    dummy_arr2[t, :, :][mask2] = 200


start_time_index = 5 
plot_time_sequence_comparison(dummy_arr1, dummy_arr2, t_start=start_time_index, num_frames=20)

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