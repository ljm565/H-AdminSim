import requests
from typing import Optional
from urllib.parse import urlencode

from h_adminsim.utils import log



class FHIRManager:
    def __init__(self, fhir_url):
        self.fhir_url = fhir_url
        
    
    def __logging(self, response: requests.Response, verbose=True) -> Optional[requests.Response]:
        """
        Log the response status code and content.

        Args:
            response (requests.Response): The HTTP response object.
            verbose (bool, optional): If True, log details. Defaults to True.

        Returns:
            Optional[requests.Response]: The response object if JSON parsing is successful, else None.
        """
        if  200 <= response.status_code < 300:
            if verbose:
                log(f'Status code: {response.status_code}', color=True)
        else:
            if verbose:
                log(f'Status code: {response.status_code}', level='error')
        
        try:
            response_json = response.json()
            if verbose:
                log(f'Response JSON: {response_json}')      
        except ValueError:
            if verbose:
                log(f'Response Text: {response.text}', level='error')
            response = None
        return response


    def create(self, 
               resource_type: str, 
               resource_data: dict, 
               headers: Optional[dict] = None, 
               verbose: bool = True) -> Optional[requests.Response]:
        """
        Create a FHIR resource of the specified type.

        Args:
            resource_type (str): FHIR resource type (e.g., "Patient", "PractitionerRole").
            resource_data (dict): FHIR resource data as a dictionary.
            headers (Optional[dict], optional): HTTP headers to use. Defaults to None.
            verbose (bool, optional): If True, log details. Defaults to True.

        Returns:
            Optional[requests.Response]: The HTTP response object if JSON parsing is successful, else None.
        """
        _id = resource_data.get('id')
        fhir_url = f'{self.fhir_url}/{resource_type}/{_id}'
        response = requests.put(
            fhir_url,
            headers={'Content-Type': 'application/fhir+json'} if headers is None else headers,
            json=resource_data,
        )

        # Log and return the response
        return self.__logging(response, verbose)
    
    
    def read(self, 
             resource_type: str, 
             id: str, 
             headers: Optional[dict] = None, 
             verbose: bool = True) -> Optional[requests.Response]:
        """
        Read a FHIR resource of the specified type and ID.

        Args:
            resource_type (str): FHIR resource type (e.g., "Patient", "PractitionerRole").
            id (str): The ID of the FHIR resource to read.
            headers (Optional[dict], optional): HTTP headers to use. Defaults to None.
            verbose (bool, optional): If True, log details. Defaults to True.

        Returns:
            Optional[requests.Response]: The HTTP response object if JSON parsing is successful, else None.
        """
        fhir_url = f'{self.fhir_url}/{resource_type}/{id}'
        response = requests.get(
            fhir_url,
            headers={'Accept': 'application/fhir+json'} if headers is None else headers,
        )

        # Log and return the response
        return self.__logging(response, verbose)
    

    def update(self, 
               resource_type: str, 
               id: str, 
               resource_data: dict, 
               headers: Optional[dict] = None,
               verbose: bool = True) -> Optional[requests.Response]:
        """
        Update a FHIR resource of the specified type and ID.

        Args:
            resource_type (str): FHIR resource type (e.g., "Patient", "PractitionerRole").
            id (str): The ID of the FHIR resource to update.
            resource_data (dict): FHIR resource data as a dictionary.
            headers (Optional[dict], optional): HTTP headers to use. Defaults to None.
            verbose (bool, optional): If True, log details. Defaults to True.

        Returns:
            Optional[requests.Response]: _description_
        """
        fhir_url = f'{self.fhir_url}/{resource_type}/{id}'
        response = requests.put(
            fhir_url,
            headers={'Content-Type': 'application/fhir+json'} if headers is None else headers,
            json=resource_data,
        )

        # Log and return the response
        return self.__logging(response, verbose)


    def delete(self, resource_type: str, id: str, verbose=True):
        fhir_url = f'{self.fhir_url}/{resource_type}/{id}'
        response = requests.delete(
            fhir_url
        )

        # Log and return the response
        return self.__logging(response, verbose)


    def read_all(self,
                 resource_type: str,
                 headers: Optional[dict] = None,
                 count: int = 100,
                 verbose: bool = True,
                 params: Optional[dict] = None) -> list[dict]:
        """
        Read all resources of a given resource type using FHIR search with optional filtering.

        Args:
            resource_type (str): FHIR resource type (e.g., "PractitionerRole").
            headers (dict, optional): HTTP headers to use.
            count (int): Number of resources to fetch per page (default: 100).
            verbose (bool): If True, log each response. Defaults to True.
            params (dict, optional): FHIR search parameters (e.g., {"specialty": "IMALL-2"}).

        Returns:
            list[dict]: List of bundle entry dicts.
        """
        all_entries = []
        headers = {'Accept': 'application/fhir+json'} if headers is None else headers

        # Build first page URL with params
        q = {'_count': count}
        if params:
            q.update({k: v for k, v in params.items() if v is not None})
        url = f"{self.fhir_url}/{resource_type}?{urlencode(q, doseq=True)}"

        while url:
            response = requests.get(url, headers=headers)
            self.__logging(response, verbose)
            try:
                bundle = response.json()
            except Exception:
                break

            if bundle.get('resourceType') != 'Bundle' or 'entry' not in bundle:
                break

            all_entries.extend(bundle['entry'])

            # Check for next link (pagination)
            next_link = next(
                (link.get('url') for link in bundle.get('link', []) if link.get('relation') == 'next'),
                None
            )
            url = next_link  # Continue if next page exists, else break

        return all_entries
    

    def delete_all(self, entry: list[dict], verbose: bool = True):
        """
        Delete all FHIR resources from a given list of resource entries.

        Args:
            entry (list[dict]): List of FHIR Bundle entries, typically from the `read_all()` method.
                                Each entry should contain a 'resource' dict with 'resourceType' and 'id'.
            verbose (bool): If True, log each deletion response. Defaults to True.
        """
        error_ids = list()

        for resource in entry:
            resource_type = resource.get('resource').get('resourceType')
            id = resource.get('resource').get('id')
            response = self.delete(resource_type, id, verbose)

            if not 200 <= response.status_code < 300:
                error_ids.append(id)

        if error_ids:
            log(f'Error(s) occurs during delete resources: {error_ids}', 'warning')
        else:
            log('Deletion successfully completed', color=True)


        

# PostgreSQL
# docker exec -it jmlee_fhir_db psql -U admin -d hapi
# SELECT * FROM hfj_resource WHERE res_type = 'Patient' LIMIT 1;
# SELECT * FROM HFJ_RES_VER WHERE RES_ID = 925754;

    