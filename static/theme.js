// Theme management
const theme = {
  init() {
    // Check for saved theme or system preference
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme) {
      document.documentElement.setAttribute("data-theme", savedTheme);
    } else if (
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: light)").matches
    ) {
      document.documentElement.setAttribute("data-theme", "light");
    } else {
      document.documentElement.setAttribute("data-theme", "dark");
    }

    // Add transition class after initial load to prevent flashing
    setTimeout(() => {
      document.body.classList.add("theme-transition");
    }, 100);
  },

  toggle() {
    const currentTheme = document.documentElement.getAttribute("data-theme");
    const newTheme = currentTheme === "light" ? "dark" : "light";

    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);

    return newTheme;
  },
};

// Initialize theme immediately to prevent flash of wrong theme
theme.init();

// Expose to window for UI interaction
window.toggleTheme = theme.toggle;
