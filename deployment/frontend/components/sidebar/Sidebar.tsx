"use client";

import { useEffect, useState } from "react";
import styles from "./Sidebar.module.css";
import { showMovil, getMovilVisible } from "@/services/movil_window_service"; // ajusta la ruta si cambia

interface SidebarProps {
  activeSection: string;
  onSectionChange: (section: string) => void;
}

interface SidebarSubSection {
  id: string;
  label: string;
  icon: string;
}

interface SidebarSection {
  id: string;
  label: string;
  icon: string;
  subSections?: SidebarSubSection[];
}

const sections: SidebarSection[] = [
  { id: "dashboard", label: "Dashboard", icon: "/assets/icons/SVG/dashboard.svg" },
  { id: "predict", label: "Predicción", icon: "/assets/icons/SVG/prediccion.svg" },
  { id: "eda", label: "Dataset", icon: "/assets/icons/SVG/lista.svg" },
  { id: "metrics", label: "Métricas", icon: "/assets/icons/SVG/metricas.svg" },
  {
    id: "wiki",
    label: "Wiki",
    icon: "/assets/icons/SVG/wiki.svg",
    subSections: [
      { id: "wiki", label: "Acerca de", icon: "/assets/icons/SVG/metricas.svg" },
      { id: "edas", label: "EDAs", icon: "/assets/icons/SVG/edalab.svg" },
      { id: "preprocesado-y-modelado", label: "Evaluación de modelos", icon: "/assets/icons/SVG/testeda.svg" },
    ],
  },
  { id: "showMovil", label: "Mostrar Movil", icon: "/assets/icons/SVG/movil.svg" },
];

export default function Sidebar({ activeSection, onSectionChange }: SidebarProps) {
  const [openMenus, setOpenMenus] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const nextOpenMenus: Record<string, boolean> = {};

    for (const section of sections) {
      if (section.subSections?.some((sub) => sub.id === activeSection)) {
        nextOpenMenus[section.id] = true;
      }
    }

    setOpenMenus((prev) => ({
      ...prev,
      ...nextOpenMenus,
    }));
  }, [activeSection]);

  const toggleMenu = (sectionId: string) => {
    setOpenMenus((prev) => ({
      ...prev,
      [sectionId]: !prev[sectionId],
    }));
  };

  return (
    <aside className={styles.sidebar__container}>
      {sections.map((section) => {
        const hasSubSections = !!section.subSections?.length;
        const isSubActive = section.subSections?.some((sub) => sub.id === activeSection);
        const isActive = activeSection === section.id || !!isSubActive;
        const isOpen = !!openMenus[section.id];

        if (hasSubSections) {
          return (
            <div key={section.id} className={styles.section}>
              <button
                onClick={() => toggleMenu(section.id)}
                className={`${isOpen ? styles.adorno_izquierda_submenu_container : ""}`}
              >
                <div className={styles.menu_item}>
                  <span
                    aria-label={section.label}
                    style={{ maskImage: `url(${section.icon})` }}
                    className={styles.menu__item}
                  />
                  <span className={styles.label + " " + styles.submenu_labels}>{section.label}</span>

                </div>

              </button>
              {isOpen && (
                <div
                  className={`${styles.submenu_container} ${isOpen ? styles.submenu_open : ""
                    }`}
                >
                  {section.subSections!.map((sub) => (
                    <button
                      key={sub.id}
                      onClick={() => onSectionChange(sub.id)}
                      className={`${activeSection === sub.id
                        ? styles.adorno_izquierda_submenu_item
                        : ""
                        }`}
                    >
                      <span
                        style={{ maskImage: `url(${sub.icon})` }}
                        className={`${styles.menu__item} ${styles.submenu_item}`}
                      />

                    </button>

                  ))}
                </div>
              )}
            </div>
          );
        }

        if (section.id === "showMovil") {
          return (
            <button
              key={section.id}
              onClick={() => showMovil()}
              className={`${styles.section} ${isActive ? styles.adorno_izquierda : ""
                } ${section.id === "showMovil" ? styles.showMovil : ""}
                  ${getMovilVisible() && section.id === "showMovil" ? styles.showMovil_True : ""} 
                `}
            >
              <span
                style={{ maskImage: `url(${section.icon})` }}
                className={`${styles.menu__item} ${styles.menu__item__normal}`}
              />

            </button>
          );
        }

        return (
          <button
            key={section.id}
            onClick={() => onSectionChange(section.id)}
            className={`${styles.section} ${isActive ? styles.adorno_izquierda : ""
              } ${section.id === "showMovil" ? styles.showMovil : ""}
                
              `}
          >
            <span
              style={{ maskImage: `url(${section.icon})` }}
              className={`${styles.menu__item} ${styles.menu__item__normal}`}
            />
            <span className={styles.label}>{section.label}</span>
          </button>
        );
      })}
    </aside>
  );
}