import { Loader2 } from "lucide-react";

const Button = ({
  children,
  variant = "primary",
  className = "",
  isLoading,
  disabled,
  ...props
}) => {
  const baseStyles =
    "inline-flex items-center justify-center rounded-lg px-6 py-3 text-sm font-medium transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none";

  const variants = {
    primary:
      "bg-[#e29578] text-[#edf6f9] hover:bg-[#006d77] hover:text-[#edf6f9] shadow-lg shadow-[#e29578]/30",
    secondary:
      "bg-[#edf6f9] text-[#006d77] border border-[#83c5be] hover:bg-[#83c5be] hover:text-[#edf6f9]",
    ghost:
      "bg-transparent text-[#006d77] hover:text-[#e29578] hover:bg-[#ffddd2]/50",
  };

  return (
    <button
      className={`${baseStyles} ${variants[variant]} ${className}`}
      disabled={isLoading || disabled}
      {...props}
    >
      {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
      {children}
    </button>
  );
};

export default Button;
