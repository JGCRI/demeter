"""
Utilities shared by the intensification and expansion algorithms.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER
"""

import os
import threading


class DiagnosticWriter:
    """Thread-safe writer for a per time step diagnostic CSV file.

    :param base_file:           Full path with file name and extension of the diagnostic file; the time step is
                                appended to the file name
    :param yr:                  Time step being processed

    """

    HEADER = 'region_id,metric_id,from_pft,to_pft,target_value\n'

    def __init__(self, base_file, yr):

        file_name, extension = os.path.splitext(base_file)
        self.path = '{0}_{1}{2}'.format(file_name, yr, extension)

        self._lock = threading.Lock()
        self._file = open(self.path, 'w')
        self._file.write(self.HEADER)

    def write(self, line):
        """Write a line; safe to call from multiple threads."""
        with self._lock:
            self._file.write(line)

    def close(self):
        """Close the file."""
        with self._lock:
            self._file.close()


def open_diagnostic(c, base_file, yr):
    """Return a DiagnosticWriter if the user selected diagnostics and outputs are written; otherwise None."""
    if c.diagnostic == 1 and c.write_outputs:
        return DiagnosticWriter(base_file, yr)

    return None
