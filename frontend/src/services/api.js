const apiUrl = import.meta.env.VITE_BACKEND_URL;

export const uploadResume = async (formDataValues) => {
  const data = new FormData();

  data.append("file", formDataValues.file);
  data.append("job_description", formDataValues.jobDescription);
  data.append("desired_role", formDataValues.role);

  const res = await fetch(`${apiUrl}/upload-resume`, {
    method: "POST",
    body: data,
  });

  if (!res.ok) {
    const errorData = await res.json();
    console.error("FastAPI Error Details:", errorData);
    throw new Error("Upload failed");
  }

  return res.json();
};
