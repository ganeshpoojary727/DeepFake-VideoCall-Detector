"use client";

import { useEffect, useState } from "react";
import { motion, useMotionValue, useSpring } from "framer-motion";

export default function CustomCursor() {
  const [isHovered, setIsHovered] = useState(false);
  const [isClicking, setIsClicking] = useState(false);
  const [isVisible, setIsVisible] = useState(false);

  const mouseX = useMotionValue(-100);
  const mouseY = useMotionValue(-100);

  // Smooth physics-based spring follow
  const springConfig = { damping: 25, stiffness: 300, mass: 0.5 };
  const smoothX = useSpring(mouseX, springConfig);
  const smoothY = useSpring(mouseY, springConfig);

  const trailingConfig = { damping: 18, stiffness: 150, mass: 0.8 };
  const trailingX = useSpring(mouseX, trailingConfig);
  const trailingY = useSpring(mouseY, trailingConfig);

  useEffect(() => {
    // Disable on touch devices
    if (window.matchMedia("(pointer: coarse)").matches) return;

    const handleMouseMove = (e: MouseEvent) => {
      mouseX.set(e.clientX);
      mouseY.set(e.clientY);
      if (!isVisible) setIsVisible(true);
    };

    const handleMouseDown = () => setIsClicking(true);
    const handleMouseUp = () => setIsClicking(false);

    const handleMouseOver = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const interactive = target.closest("button, a, input, [role='button'], .glass-panel-interactive, .cursor-pointer");
      setIsHovered(!!interactive);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("mouseover", handleMouseOver);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("mouseover", handleMouseOver);
    };
  }, [mouseX, mouseY, isVisible]);

  if (!isVisible) return null;

  return (
    <div className="pointer-events-none fixed inset-0 z-50 overflow-hidden">
      {/* Outer Glowing Trailing Ring / Spotlight */}
      <motion.div
        className="absolute rounded-full border border-sky-400/40 bg-sky-500/10 backdrop-blur-[1px]"
        style={{
          x: trailingX,
          y: trailingY,
          translateX: "-50%",
          translateY: "-50%",
        }}
        animate={{
          width: isHovered ? 64 : isClicking ? 28 : 40,
          height: isHovered ? 64 : isClicking ? 28 : 40,
          borderColor: isHovered ? "rgba(56, 189, 248, 0.9)" : "rgba(56, 189, 248, 0.4)",
          boxShadow: isHovered
            ? "0 0 35px rgba(56, 189, 248, 0.5), inset 0 0 15px rgba(56, 189, 248, 0.3)"
            : "0 0 15px rgba(56, 189, 248, 0.2)",
        }}
        transition={{ type: "spring", stiffness: 350, damping: 25 }}
      />

      {/* Inner Precision Neon Dot */}
      <motion.div
        className="absolute rounded-full bg-sky-400 shadow-[0_0_12px_#38bdf8]"
        style={{
          x: smoothX,
          y: smoothY,
          translateX: "-50%",
          translateY: "-50%",
        }}
        animate={{
          scale: isClicking ? 0.6 : isHovered ? 1.6 : 1,
          backgroundColor: isHovered ? "#38bdf8" : "#ffffff",
        }}
        transition={{ duration: 0.15 }}
      >
        <div className="h-2 w-2 rounded-full" />
      </motion.div>
    </div>
  );
}
