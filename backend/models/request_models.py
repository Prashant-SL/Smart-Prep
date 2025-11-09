from pydantic import BaseModel, Field

class ResumeTextRequest(BaseModel):
    resume_text: str = Field(..., description="Raw text content of the resume")
    job_description: str = Field(..., description="Job description for the target role")
    desired_role: str = Field(..., description="The target role user wants to apply for")