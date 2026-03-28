const Card = ({ children, className = "" }) => (
  <div
    className={`rounded-xl border border-[#83c5be] bg-[#edf6f9] shadow-xl ${className}`}
  >
    {children}
  </div>
);

export default Card;
