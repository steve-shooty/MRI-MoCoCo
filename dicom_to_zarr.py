import os
import pydicom
import numpy as np
import zarr

def convert_dicom_to_zarr(dicom_dir: str, zarr_path: str):
    """
    Reads a DICOM-Series from a folder, convert them in a 3D-array and save the Zarr-file aas well as metadata.
    Liest eine DICOM-Serie aus einem Ordner, konvertiert sie in ein 3D-Array 
    und speichert es zusammen mit Metadaten in einer Zarr-Datei.
    
    Args:
        dicom_dir (str): Pfad zum Ordner mit den DICOM-Dateien.
        zarr_path (str): Pfad, unter dem die .zarr-Datei gespeichert werden soll.
    """
    print(f"Lese DICOM-Dateien aus: {dicom_dir}")
    
    slices = []
    for f in sorted(os.listdir(dicom_dir)):
        file_path = os.path.join(dicom_dir, f)
        try:
            dcm = pydicom.dcmread(file_path)
            if 'PixelData' in dcm:
                slices.append(dcm)
        except pydicom.errors.InvalidDicomError:
            print(f"Warnung: {f} ist keine gültige DICOM-Datei und wird übersprungen.")
            continue
            
    if not slices:
        print("Fehler: Keine gültigen DICOM-Dateien im Verzeichnis gefunden.")
        return

    slices.sort(key=lambda x: int(x.InstanceNumber))
    
    print("Stapele DICOM-Schichten zu einem 3D-Raum...")
    pixel_array = np.stack([s.pixel_array for s in slices])

    first_slice = slices[0]
    metadata = {
        'PixelSpacing': [float(x) for x in first_slice.PixelSpacing],
        'SliceThickness': float(first_slice.SliceThickness),
        'Rows': int(first_slice.Rows),
        'Columns': int(first_slice.Columns),
        'PatientID': str(first_slice.PatientID),
    }
    print(f"Extrahierte Metadaten: {metadata}")

    print(f"Speichere Array der Größe {pixel_array.shape} in {zarr_path}...")
    chunks = (64, 128, 128) 

    z_array = zarr.open(zarr_path, mode='w', shape=pixel_array.shape, 
                        chunks=chunks, dtype=pixel_array.dtype)
    z_array[:] = pixel_array

    z_array.attrs['metadata'] = metadata
    
    print("Konvertierung abgeschlossen!")