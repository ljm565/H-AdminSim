import requests

from utils import log



class FHIRManager:
    def __init__(self, config):
        self.fhir_url = config.fhir_url
        
    
    def __logging(self, response):
        if  200 <= response.status_code < 300:
            log(f'Status code: {response.status_code}', color=True)
        else:
            log(f'Status code: {response.status_code}', level='error')
        
        try:
            response_json = response.json()
            log(f'Response JSON: {response_json}')      
        except ValueError:
            log(f'Response Text: {response.text}', level='error')
            response = None
        return response


    def create(self, resource_type: str, resource_data: dict, headers=None):
        _id = resource_data.get('id')
        fhir_url = f'{self.fhir_url}/{resource_type}/{_id}'
        response = requests.put(
            fhir_url,
            headers={'Content-Type': 'application/fhir+json'} if headers is None else headers,
            json=resource_data,
        )

        # Log and return the response
        return self.__logging(response)
    
    
    def read(self, resource_type: str, id: str, headers=None):
        fhir_url = f'{self.fhir_url}/{resource_type}/{id}'
        response = requests.get(
            fhir_url,
            headers={'Accept': 'application/fhir+json'} if headers is None else headers,
        )

        # Log and return the response
        return self.__logging(response)
    

    def update(self, resource_type: str, id: str, resource_data, headers=None):
        fhir_url = f'{self.fhir_url}/{resource_type}/{id}'
        response = requests.put(
            fhir_url,
            headers={'Content-Type': 'application/fhir+json'} if headers is None else headers,
            json=resource_data,
        )

        # Log and return the response
        return self.__logging(response)


    def delete(self, resource_type: str, id: str):
        fhir_url = f'{self.fhir_url}/{resource_type}/{id}'
        response = requests.delete(
            fhir_url
        )

        # Log and return the response
        return self.__logging(response)




        

# PostgreSQL
# docker exec -it jmlee_fhir_db psql -U admin -d hapi
# SELECT * FROM hfj_resource WHERE res_type = 'Patient' LIMIT 1;
# SELECT * FROM HFJ_RES_VER WHERE RES_ID = 925754;

    