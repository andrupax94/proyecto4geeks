"use client";
import { motion } from "framer-motion";
import { ReactNode } from "react";

interface AnimatedChartProps {
  children: ReactNode;
  isVisible: boolean;
}

export default function AnimatedChart({ children, isVisible }: AnimatedChartProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={isVisible ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="w-full"
    >
      {children}
    </motion.div>
  );
}
