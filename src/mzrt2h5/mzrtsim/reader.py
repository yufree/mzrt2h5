import xml.etree.ElementTree as ET
import base64
import zlib
import numpy as np
import re

class SimpleMzMLReader:
    def __init__(self, filename):
        self.filename = filename
        self.ns = {'mzml': "http://psi.hupo.org/ms/mzml"}

    def _decode_binary(self, binary_element, array_length, cv_params):
        if binary_element.text is None:
            return np.array([])
        
        encoded_data = binary_element.text.strip()
        if not encoded_data:
            return np.array([])
            
        decoded = base64.b64decode(encoded_data)
        
        # Check compression
        is_zlib = 'MS:1000574' in cv_params
        if is_zlib:
            try:
                decoded = zlib.decompress(decoded)
            except Exception as e:
                # If it's supposed to be zlib but fails, that's a critical error
                raise ValueError(f"Zlib decompression failed: {e}")

        # Check precision
        dtype = np.float32
        if 'MS:1000523' in cv_params: # 64-bit float
            dtype = np.float64
        elif 'MS:1000521' in cv_params: # 32-bit float
            dtype = np.float32
        elif 'MS:1000522' in cv_params: # 64-bit integer
            dtype = np.int64
        elif 'MS:1000519' in cv_params: # 32-bit integer
            dtype = np.int32

        return np.frombuffer(decoded, dtype=dtype)

    def get_spectra(self):
        """
        Yields dictionaries containing:
        - rt: retention time in seconds
        - mz: m/z array
        - intensity: intensity array
        - id: spectrum id
        - ms_level: ms level
        """
        context = ET.iterparse(self.filename, events=('end',))
        
        for event, elem in context:
            if elem.tag.endswith('spectrum'):
                spectrum = {}
                spectrum['id'] = elem.attrib.get('id')
                spectrum['index'] = elem.attrib.get('index')
                default_array_length = int(elem.attrib.get('defaultArrayLength', 0))
                
                # Get CV params
                spec_cv_params = elem.findall('mzml:cvParam', self.ns)
                
                # MS Level
                ms_level = 1
                for cv in spec_cv_params:
                    if cv.attrib.get('accession') == 'MS:1000511':
                        ms_level = int(cv.attrib.get('value'))
                        break
                spectrum['ms_level'] = ms_level

                # Scan List for RT
                scan_list = elem.find('mzml:scanList', self.ns)
                rt = 0.0
                if scan_list is not None:
                    scan = scan_list.find('mzml:scan', self.ns)
                    if scan is not None:
                        for cv in scan.findall('mzml:cvParam', self.ns):
                            if cv.attrib.get('accession') == 'MS:1000016':
                                val = float(cv.attrib.get('value'))
                                unit = cv.attrib.get('unitName')
                                if unit == 'minute':
                                    val *= 60
                                rt = val
                                break
                spectrum['rt'] = rt
                
                # Binary Data
                binary_list = elem.find('mzml:binaryDataArrayList', self.ns)
                mz_array = np.array([])
                int_array = np.array([])
                
                if binary_list is not None:
                    for binary_array in binary_list.findall('mzml:binaryDataArray', self.ns):
                        cvs = [x.attrib.get('accession') for x in binary_array.findall('mzml:cvParam', self.ns)]
                        data = self._decode_binary(binary_array.find('mzml:binary', self.ns), default_array_length, cvs)
                        
                        if 'MS:1000514' in cvs: # m/z array
                            mz_array = data
                        elif 'MS:1000515' in cvs: # intensity array
                            int_array = data
                            
                spectrum['mz'] = mz_array
                spectrum['intensity'] = int_array
                
                yield spectrum
                
                # Cleanup to save memory
                elem.clear()

    def get_rts(self):
        """
        Quickly extract all retention times.
        """
        rts = []
        context = ET.iterparse(self.filename, events=('end',))
        for event, elem in context:
            if elem.tag.endswith('spectrum'):
                # Simplified extraction just for RT
                scan_list = elem.find('mzml:scanList', self.ns)
                if scan_list is not None:
                    scan = scan_list.find('mzml:scan', self.ns)
                    if scan is not None:
                        for cv in scan.findall('mzml:cvParam', self.ns):
                            if cv.attrib.get('accession') == 'MS:1000016':
                                val = float(cv.attrib.get('value'))
                                unit = cv.attrib.get('unitName')
                                if unit == 'minute':
                                    val *= 60
                                rts.append(val)
                                break
                elem.clear()
        return np.array(rts)
