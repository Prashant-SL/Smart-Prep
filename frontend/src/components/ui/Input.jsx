import { AlertCircle } from "lucide-react";

const Input = ({ label, id, error, className = "", ...props }) => (
  <div className={`space-y-2 ${className}`}>
    {label && (
      <label htmlFor={id} className="text-sm font-medium text-[#006d77]">
        {label}
      </label>
    )}
    <input
      id={id}
      className={`flex h-11 w-full rounded-lg border border-[#83c5be] bg-[#edf6f9] px-3 py-2 text-sm text-[#006d77] placeholder:text-[#83c5be] focus:border-[#e29578] focus:outline-none focus:ring-1 focus:ring-[#e29578] disabled:cursor-not-allowed disabled:opacity-50 transition-colors`}
      {...props}
    />
    {error && (
      <p className="text-xs text-[#e29578] flex items-center gap-1">
        <AlertCircle className="w-3 h-3" /> {error}
      </p>
    )}
  </div>
);

export default Input;
