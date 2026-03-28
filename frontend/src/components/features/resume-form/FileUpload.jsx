import { useRef } from "react";
import { FileText, X, Upload, AlertCircle } from "lucide-react";

const FileUpload = ({ file, onFileChange, error }) => {
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type === "application/pdf") {
      onFileChange(droppedFile);
    } else {
      // Handle non-pdf drop
    }
  };

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-[#006d77]">Resume (PDF)</label>
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`
          relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-all cursor-pointer
          ${error ? "border-[#e29578] bg-[#ffddd2]/50" : file ? "border-[#006d77] bg-[#edf6f9]" : "border-[#83c5be] hover:border-[#e29578] hover:bg-[#ffddd2]/30"}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={(e) => e.target.files[0] && onFileChange(e.target.files[0])}
        />

        {file ? (
          <div className="flex items-center gap-3 text-[#006d77]">
            <FileText className="h-8 w-8 text-[#006d77]" />
            <div className="text-center">
              <p className="text-sm font-medium">{file.name}</p>
              <p className="text-xs text-[#83c5be]">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onFileChange(null);
              }}
              className="ml-2 rounded-full p-1 hover:bg-[#83c5be]"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        ) : (
          <div className="text-center">
            <Upload className="mx-auto h-10 w-10 text-[#83c5be] mb-3" />
            <p className="text-sm text-[#006d77]">
              <span className="font-semibold text-[#e29578]">
                Click to upload
              </span>{" "}
              or drag and drop
            </p>
            <p className="text-xs text-[#83c5be] mt-1">PDF only (max 5MB)</p>
          </div>
        )}
      </div>
      {error && (
        <p className="text-xs text-[#e29578] flex items-center gap-1">
          <AlertCircle className="w-3 h-3" /> {error}
        </p>
      )}
    </div>
  );
};

export default FileUpload;
