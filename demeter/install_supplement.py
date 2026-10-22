import logging
import os
import zipfile
from importlib.metadata import version
from io import BytesIO

import requests

logger = logging.getLogger('demeter_runtime')


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

    # seconds to wait for the data server to respond
    REQUEST_TIMEOUT_SECONDS = 300

    def __init__(self, example_data_directory):

        # full path to the root directory where the example dir will be stored
        self.example_data_directory = example_data_directory

    def fetch_zenodo(self):
        """Download and unpack the Zenodo example data supplement for the
        current distribution."""

        # get the current version of that is installed
        current_version = version('demeter')

        try:
            data_link = InstallSupplement.DATA_VERSION_URLS[current_version]

        except KeyError:
            # fall back to the most recent data supplement when none is registered for this version
            latest = list(InstallSupplement.DATA_VERSION_URLS)[-1]
            logger.warning(f"No example data registered for Demeter version {current_version}; using the data "
                           f"supplement for version {latest}.")
            data_link = InstallSupplement.DATA_VERSION_URLS[latest]

        # retrieve content from URL
        logger.info("Downloading example data for Demeter version {}...".format(current_version))
        r = requests.get(data_link, timeout=InstallSupplement.REQUEST_TIMEOUT_SECONDS)
        r.raise_for_status()

        with zipfile.ZipFile(BytesIO(r.content)) as zipped:

            # extract each file in the zipped dir to the project
            for f in zipped.namelist():
                logger.info("Unzipped: {}".format(os.path.join(self.example_data_directory, f)))
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
