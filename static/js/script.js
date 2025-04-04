document.addEventListener('DOMContentLoaded', function() {
    // Navbar scrolling effect
    window.addEventListener('scroll', function() {
        const navbar = document.getElementById('navbar');
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // Mobile menu toggle
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    if (hamburger) {
        hamburger.addEventListener('click', function() {
            navMenu.classList.toggle('active');
        });
    }

    // Close menu when clicking outside
    document.addEventListener('click', function(event) {
        if (!event.target.closest('.nav-menu') && !event.target.closest('.hamburger')) {
            if (navMenu.classList.contains('active')) {
                navMenu.classList.remove('active');
            }
        }
    });
});

// Function to open report document
function openReportDoc() {
    window.open('/static/assets/Report.pdf', '_blank');
}

// Function to open research paper
function openResearchPDF() {
    window.open('/static/assets/Research_paper.pdf', '_blank');
}
