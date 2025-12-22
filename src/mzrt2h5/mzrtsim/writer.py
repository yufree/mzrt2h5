import base64
import struct
import zlib
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np

def write_mzml(filename, spectra_list):
    """
    Writes a list of spectra to an mzML file.
    
    spectra_list: list of dicts, each containing:
      'mz': numpy array of m/z
      'intensity': numpy array of intensity
      'rt': retention time in seconds
      'id': scan id (optional)
      'ms_level': ms level (default 1)
    """
    
    mzml = ET.Element('mzML', xmlns="http://psi.hupo.org/ms/mzml", version="1.1.0")
    cvList = ET.SubElement(mzml, 'cvList', count="2")
    ET.SubElement(cvList, 'cv', id="MS", fullName="Proteomics Standards Initiative Mass Spectrometry Ontology", version="4.1.7", URI="https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo")
    ET.SubElement(cvList, 'cv', id="UO", fullName="Unit Ontology", version="14:07:2009", URI="http://obo.cvs.sourceforge.net/*checkout*/obo/obo/ontology/phenotype/unit.obo")
    
    run = ET.SubElement(mzml, 'run', id="run_1", defaultInstrumentConfigurationRef="IC1", startTimeStamp="2023-01-01T00:00:00Z")
    spectrumList = ET.SubElement(run, 'spectrumList', count=str(len(spectra_list)), defaultDataProcessingRef="DP1")
    
    for i, s in enumerate(spectra_list):
        mz_array = s['mz']
        int_array = s['intensity']
        rt = s.get('rt', 0.0)
        scan_id = s.get('id', f"scan={i+1}")
        ms_level = s.get('ms_level', 1)
        
        spectrum = ET.SubElement(spectrumList, 'spectrum', id=scan_id, index=str(i), defaultArrayLength=str(len(mz_array)))
        ET.SubElement(spectrum, 'cvParam', cvRef="MS", accession="MS:1000511", name="ms level", value=str(ms_level))
        ET.SubElement(spectrum, 'cvParam', cvRef="MS", accession="MS:1000579", name="MS1 spectrum", value="")
        
        scanList = ET.SubElement(spectrum, 'scanList', count="1")
        ET.SubElement(scanList, 'cvParam', cvRef="MS", accession="MS:1000795", name="no combination", value="")
        scan = ET.SubElement(scanList, 'scan')
        ET.SubElement(scan, 'cvParam', cvRef="MS", accession="MS:1000016", name="scan start time", value=str(rt), unitCvRef="UO", unitAccession="UO:0000010", unitName="second")
        
        binaryDataArrayList = ET.SubElement(spectrum, 'binaryDataArrayList', count="2")
        
        # M/Z Array
        _add_binary_array(binaryDataArrayList, mz_array, "m/z array", "MS:1000514", "MS:1000040")
        
        # Intensity Array
        _add_binary_array(binaryDataArrayList, int_array, "intensity array", "MS:1000515", "MS:1000040")

    tree = ET.ElementTree(mzml)
    # Pretty print hack
    xmlstr = minidom.parseString(ET.tostring(mzml)).toprettyxml(indent="  ")
    with open(filename, "w") as f:
        f.write(xmlstr)

def _add_binary_array(parent, array, name, accession, unit_accession=None):
    binaryDataArray = ET.SubElement(parent, 'binaryDataArray', encodedLength="0")
    ET.SubElement(binaryDataArray, 'cvParam', cvRef="MS", accession="MS:1000523", name="64-bit float", value="")
    ET.SubElement(binaryDataArray, 'cvParam', cvRef="MS", accession="MS:1000574", name="zlib compression", value="")
    ET.SubElement(binaryDataArray, 'cvParam', cvRef="MS", accession=accession, name=name, value="", unitCvRef="MS" if unit_accession else None, unitAccession=unit_accession, unitName="m/z" if "m/z" in name else "number of counts")
    
    # Encode
    if len(array) > 0:
        data = array.astype(np.float64).tobytes()
        compressed = zlib.compress(data)
        encoded = base64.b64encode(compressed).decode('utf-8')
        
        binary = ET.SubElement(binaryDataArray, 'binary')
        binary.text = encoded
        binaryDataArray.set("encodedLength", str(len(encoded)))
    else:
        binary = ET.SubElement(binaryDataArray, 'binary')
        binary.text = ""

