from typing import Any



class Hospital:
    def __init__(self, 
                 hospital_name: str,
                 department_num: int,
                 doctor_num: int,
                 time: dict,
                 **kwargs: Any):
        """
        Hospital metadata with its departments.

        Args:
            hospital_name (str): Hospital name.
            department_num (int): Number of departments.
            doctor_num (int): Number of doctors.
            time (dict): Operating-time settings.
            **kwargs (Any): Extra metadata, set as attributes.
        """
        self.hospital_name = hospital_name
        self.department_num = department_num
        self.doctor_num = doctor_num
        self.time = time
        self.department: list[Department] = []
        for key, value in kwargs.items():
            setattr(self, key, value)


    def add_department(self, department_name: str, **kwargs: Any) -> 'Department':
        """
        Add a department to the hospital.

        Args:
            department_name (str): Name of the department to add.
            **kwargs (Any): Department attributes.
        
        Returns:
            Department: The newly created Department object.
        """
        dept = Department(department_name, **kwargs)
        self.department.append(dept)
        return dept


    def reset_departments(self) -> None:
        """
        Remove all departments from the hospital.
        """
        self.department = []


    def __repr__(self) -> str:
        return f"Hospital(name={self.hospital_name}, departments={[d.name for d in self.department]}, time={self.time})"



class Department:
    def __init__(self, name: str, **kwargs: Any):
        """
        Hospital department with its doctors.

        Args:
            name (str): Department name.
            **kwargs (Any): Extra attributes.
        """
        self.name = name
        self.doctor: list[Doctor] = []
        for key, value in kwargs.items():
            setattr(self, key, value)


    def add_doctor(self, doctor_name: str, **kwargs: Any) -> 'Doctor':
        """
        Add a doctor to the department.
        
        Args:
            doctor_name (str): Name of the doctor to add.
            **kwargs (Any): Doctor attributes.
        
        Returns:
            Doctor: The newly created Doctor object.
        """
        doctor = Doctor(doctor_name, self, **kwargs)
        self.doctor.append(doctor)
        return doctor

    
    def reset_doctors(self) -> None:
        """
        Remove all doctors from the department.
        """
        self.doctor = []


    def __repr__(self) -> str:
        return f"Department(name={self.name}, doctors={[d.name for d in self.doctor]})"



class Doctor:
    def __init__(self, name: str, department: Department, **kwargs: Any):
        """
        Doctor with a schedule and patients.

        Args:
            name (str): Doctor name.
            department (Department): The doctor's department.
            **kwargs (Any): Extra attributes.
        """
        self.name = name
        self.department = department
        self.schedule = []
        self.patient: list[Patient] = []
        for key, value in kwargs.items():
            setattr(self, key, value)

    
    def add_patient(self, patient_name: str, **kwargs: Any) -> 'Patient':
        """
        Add a patient to the doctor.
        
        Args:
            patient_name (str): Name of the patient to add.
            **kwargs (Any): Patient attributes.
        
        Returns:
            Patient: The newly created Patient object.
        """
        patient = Patient(patient_name, self, **kwargs)
        self.patient.append(patient)
        return patient

    
    def reset_patients(self) -> None:
        """
        Remove all patients from the doctor.
        """
        self.patient = []


    def __repr__(self) -> str:
        return f"Doctor(name={self.name}, department={self.department.name}), schedule={self.schedule})"
    


class Patient:
    def __init__(self, name: str, attending_physician: Doctor, **kwargs: Any):
        """
        Patient with an attending physician and a schedule.

        Args:
            name (str): Patient name.
            attending_physician (Doctor): The attending physician.
            **kwargs (Any): Extra attributes.
        """
        self.name = name
        self.attending_physician = attending_physician
        self.schedule = []
        for key, value in kwargs.items():
            setattr(self, key, value)
    

    def __repr__(self) -> str:
        return f"Patient(name={self.name}, department={self.attending_physician.department.name}), attending_physician={self.attending_physician.name}), schedule={self.schedule})"
