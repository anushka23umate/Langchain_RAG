from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: Optional[int] = None  #even when this parameter is not provided, pydantic will not raise an error
    email: Optional[EmailStr] = None #pydantic will validate the email format
    cgpa: float = Field(ge=0.0, le=10.0, default=5)  # cgpa must be between 0.0 and 10.0, we have used Field to set constraints

new_student = {'name':'nitish', 'email':"anushka@gmail.com"}  
student = Student(**new_student)

# print(type(student))
# print(student)  


#convert pydantic model to dictionary
# pydantic models can be converted to dictionary using the dict() function
student_dict = dict(student)  # Convert to dictionary
print(student_dict)  # {'name': 'nitish', 'age': None, 'email': '


student_json = student.model_dump_json()  # Convert to JSON string
print(student_json)  # '{"name": "nitish", "age": null, "email": "