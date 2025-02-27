import os
import zipfile

import requests

from pkg_resources import get_distribution
from io import BytesIO as BytesIO


class InstallSupplement:
    """Download and unpack example data supplement from Zenodo that matches the current installed
    distribution.
    :param example_data_directory:              Full path to the directory you wish to install
                                                the example data to.  Must be write-enabled
                                                for the user.
    """

    # URL for DOI minted example data hosted on Zenodo
    DATA_VERSION_URLS = {
        '1.3.1': 'https://zenodo.org/record/7240315/files/config_gcam_reference.zip?download=1',
        '2.0.0': 'https://zenodo.org/record/7240315/files/config_gcam_reference.zip?download=1',
        '2.0.1': 'https://zenodo.org/record/7240315/files/config_gcam_reference.zip?download=1',
    }

    def __init__(self, example_data_directory):

        # full path to the root directory where the example dir will be stored
        self.example_data_directory = example_data_directory

    def fetch_zenodo(self):
        """Download and unpack the Zenodo example data supplement for the
        current distribution."""

        # get the current version of that is installed
        current_version = get_distribution('demeter').version

        try:
            data_link = InstallSupplement.DATA_VERSION_URLS[current_version]

        except KeyError:
            msg = "Link to data missing for current version:  {}.  Please contact admin."
            raise(msg.format(current_version))

        # retrieve content from URL
        print("Downloading example data for Demeter version {}...".format(current_version))
        r = requests.get(data_link)

        with zipfile.ZipFile(BytesIO(r.content)) as zipped:

            # extract each file in the zipped dir to the project
            for f in zipped.namelist():
                print("Unzipped: {}".format(os.path.join(self.example_data_directory, f)))
                zipped.extract(f, self.example_data_directory)


def get_package_data(example_data_directory):
    """Download and unpack example data supplement from Zenodo that matches the current installed
    distribution.

    :param example_data_directory:              Full path to the directory you wish to install
                                                the example data to.  Must be write-enabled
                                                for the user.

    :type example_data_directory:               str

    """

    zen = InstallSupplement(example_data_directory)

    zen.fetch_zenodo()
