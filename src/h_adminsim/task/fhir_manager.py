import requests

from h_adminsim.utils import log



class FHIRManager:
    def __init__(self, config):
        self.fhir_url = config.fhir_url
        
    
    def __logging(self, response, verbose=True):
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


    def create(self, resource_type: str, resource_data: dict, headers=None, verbose=True):
        _id = resource_data.get('id')
        fhir_url = f'{self.fhir_url}/{resource_type}/{_id}'
        response = requests.put(
            fhir_url,
            headers={'Content-Type': 'application/fhir+json'} if headers is None else headers,
            json=resource_data,
        )

        # Log and return the response
        return self.__logging(response, verbose)
    
    
    def read(self, resource_type: str, id: str, headers=None, verbose=True):
        fhir_url = f'{self.fhir_url}/{resource_type}/{id}'
        response = requests.get(
            fhir_url,
            headers={'Accept': 'application/fhir+json'} if headers is None else headers,
        )

        # Log and return the response
        return self.__logging(response, verbose)
    

    def update(self, resource_type: str, id: str, resource_data, headers=None, verbose=True):
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


    def read_all(self, resource_type: str, headers=None, count=100, verbose=True):
        """
        Read all resources of a given resource type using FHIR search.

        Args:
            resource_type (str): The type of the FHIR resource (e.g., "Patient").
            headers (dict, optional): HTTP headers to use.
            count (int): Number of resources to fetch per page (default: 100).
            verbose (bool): If True, log each deletion response. Defaults to True.

        Returns:
            list: List of resource entries.
        """
        all_entries = []
        url = f'{self.fhir_url}/{resource_type}?_count={count}'
        headers = {'Accept': 'application/fhir+json'} if headers is None else headers

        while url:
            response = requests.get(url, headers=headers)
            bundle = response.json()
            self.__logging(response, verbose)
            
            if bundle.get('resourceType') != 'Bundle' or 'entry' not in bundle:
                break

            all_entries.extend(bundle['entry'])

            # Check for next link (pagination)
            next_link = next(
                (link['url'] for link in bundle.get('link', []) if link['relation'] == 'next'),
                None
            )
            url = next_link  # Continue if next page exists, else break

        return all_entries
    

    def delete_all(self, entry: list[dict], verbose=True):
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

    