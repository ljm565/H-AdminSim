class Hospital:
    def __init__(self, name: str, time: dict):
        self.name = name
        self.time = time
        self.department: list[Department] = []


    def add_department(self, department_name: str):
        """
        Add a department to the hospital.

        Args:
            department_name (str): Name of the department to add.
        
        Returns:
            Department: The newly created Department object.
        """
        dept = Department(department_name)
        self.department.append(dept)
        return dept


    def reset_departments(self):
        """
        Reset the list of departments in the hospital.
        """
        self.department = []


    def __repr__(self):
        return f"Hospital(name={self.name}, departments={[d.name for d in self.department]}, time={self.time})"



class Department:
    def __init__(self, name: str):
        self.name = name
        self.doctor: list[Doctor] = []


    def add_doctor(self, doctor_name: str, schedule):
        """
        Add a doctor to the department.
        
        Args:
            doctor_name (str): Name of the doctor to add.
            schedule: TODO
        
        Returns:
            Doctor: The newly created Doctor object.
        """
        doctor = Doctor(doctor_name, self, schedule)
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
    def __init__(self, name: str, department: Department, schedule):
        self.name = name
        self.department = department
        self.schedule = schedule


    def __repr__(self):
        return f"Doctor(name={self.name}, department={self.department.name}), schedule={self.schedule})"
