from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional, Union

import pybullet
from pybullet_utils import bullet_client as bc

# ----------------------------------------------------------------------------
# Connect to the PyBullet simulator using different methods.
# ----------------------------------------------------------------------------

class connection_types(Enum):
    DIRECT = pybullet.DIRECT
    GUI = pybullet.GUI
    GUI_SERVER = pybullet.GUI_SERVER
    SERVER = pybullet.SHARED_MEMORY_SERVER
    SHARED_MEM = pybullet.SHARED_MEMORY
    TCP = pybullet.TCP


class pybullet_connection(ABC):

    @abstractmethod
    def __init__(self, connection_type, options):
        pass

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def close(self):
        pass


class single_connection(pybullet_connection):
    """
    Create a bullet simulation with a single connection.

    keyword args:
    -------------
    connection_type: 'direct'/'DIRECT' for headless connection or 
        'gui'/'GUI' for a graphical simulator (useful for debugging).
    options: command-line options for the GUI server. See PyBullet 
        docs for details. Ignored if connection type is 'direct'.
    """
    def __init__(self, connection_type:str='direct', 
                 options:Optional[str]=None):

        if not connection_type in ('direct', 'gui', 'DIRECT', 'GUI'):
            raise ValueError(f"'{connection_type}' connection is invalid." 
                + " Connection type must be either 'direct' or 'gui'.")
        self.connection_type = (
            connection_types.DIRECT if connection_type in ('DIRECT', 'direct')
            else connection_types.GUI
        )
        self.options = options if options else ''
        self._phys_client = None

    @property
    def client(self):
        return self._phys_client

    @property
    def client_id(self) -> int:
        if self._phys_client and (self._phys_client._client >= 0):
            return self._phys_client._client
        else:
            print(
                "No client ID because physics server is not connected. " 
                + "Use the `.connect()` method to start a new server first."
            )
    
    def connect(self):
        """Start the physics server."""
        self._phys_client = bc.BulletClient(
            connection_mode=self.connection_type.value, options=self.options
        )

    def close(self):
        self._phys_client.disconnect()
        self._phys_client = None


class multiclient_server(pybullet_connection):
    """
    Set up a shared memory physics server with multiple clients.

    keyword args:
    -------------
    connection_type: 'shared_memory'/'SHARED_MEMORY' for headless 
        connection or 'gui_server'/'GUI_SERVER' for a graphical 
        simulator (useful for debugging).
    options: command-line options for the GUI server. See PyBullet 
        docs for details. Ignored if connection type is 'direct'.
    """
    def __init__(self, connection_type:str='server', 
                 options:Optional[str]=None):

        if not connection_type in (
            'server', 'SERVER', 'gui_server', 'GUI_SERVER'
            ):
            raise ValueError(
                f"'{connection_type}' connection is invalid. Connection type" 
                + " must be either 'shared_memory' or 'gui_server'.")
        self.connection_type = (
            connection_types.SERVER if connection_type in ('server', 'SERVER')
            else connection_types.GUI_SERVER
        )
        self.options = options if options else ''
        self._phys_server = None
        self._phys_clients = {}

    @property
    def server(self):
        return self._phys_server

    @property
    def server_id(self):
        if not self._phys_server:
            return (
                "No connection to the PyBullet physics server. " 
                + "Connect to server first using the `connect()` method."
            )
        return self._phys_server._client

    @property
    def num_clients(self):
        return len(self._phys_clients)
    
    @property
    def client_ids(self) -> List[int]:
        if not self._phys_clients:
            return (
                "No clients found. Use `add_client()` to add new clients."
            )
        return list(self._phys_clients.keys())

    def get_client(self, ID:int):
        return self._phys_clients.get(ID)

    def connect(self):
        """Start the physics server."""
        if (
            (self.options) 
            and 
            (self.connection_type == connection_types.GUI_SERVER)
            ):
            self._phys_server = bc.BulletClient(
                connection_mode=self.connection_type.value, options=self.options
            )
        else:
            self._phys_server = bc.BulletClient(
                connection_mode=self.connection_type.value
            )

    def add_client(self):
        """Connect a new client to the physics server."""
        if not self._phys_server:
            return (
                "No physics server running. Use `connect()` to start a physics" 
                + " server before adding clients."
            )
        phys_client = bc.BulletClient(
            connection_mode=connection_types.SHARED_MEM.value
        )
        self._phys_clients[phys_client._client] = phys_client

    def remove_client(self, client_id):
        """Disconnect a client and remove it from the dict."""
        if not client_id in self._phys_clients.keys():
            raise KeyError(
                f"'{client_id}' is not a valid client ID."
            )
        client = self._phys_clients.get(client_id)
        client.disconnect()
        if not client.isConnected():
            _ = self._phys_clients.pop(client_id, None)
            del client

    def close(self):
        """Shuts down the physics server and all clients."""
        self._phys_server.disconnect()
        self._phys_server = None
        self._phys_clients = {}
