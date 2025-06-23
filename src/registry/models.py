class Hospital:
    def __init__(self, 
                 hospital_name: str,
                 department_num: int,
                 doctor_num: int,
                 time: dict,
                 **kwargs):
        self.hospital_name = hospital_name
        self.department_num = department_num
        self.doctor_num = doctor_num
        self.time = time
        self.department: list[Department] = []
        for key, value in kwargs.items():
            setattr(self, key, value)


    def add_department(self, department_name: str, **kwargs):
        """
        Add a department to the hospital.

        Args:
            department_name (str): Name of the department to add.
        
        Returns:
            Department: The newly created Department object.
        """
        dept = Department(department_name, **kwargs)
        self.department.append(dept)
        return dept


    def reset_departments(self):
        """
        Reset the list of departments in the hospital.
        """
        self.department = []


    def __repr__(self):
        return f"Hospital(name={self.hospital_name}, departments={[d.name for d in self.department]}, time={self.time})"



class Department:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.doctor: list[Doctor] = []
        for key, value in kwargs.items():
            setattr(self, key, value)


    def add_doctor(self, doctor_name: str, **kwargs):
        """
        Add a doctor to the department.
        
        Args:
            doctor_name (str): Name of the doctor to add.
            schedule (list[list[float]]): Doctor's fixed schedule time.
        
        Returns:
            Doctor: The newly created Doctor object.
        """
        doctor = Doctor(doctor_name, self, **kwargs)
        self.doctor.append(doctor)
        return doctor

    
    def reset_doctors(self):
        """
        Reset the list of doctors in the department.
        """
        self.doctor = []


    def __repr__(self):
        return f"Department(name={self.name}, doctors={[d.name for d in self.doctor]})"



class Doctor:
    def __init__(self, name: str, department: Department, **kwargs):
        self.name = name
        self.department = department
        for key, value in kwargs.items():
            setattr(self, key, value)


    def __repr__(self):
        return f"Doctor(name={self.name}, department={self.department.name}), schedule={self.schedule})"
