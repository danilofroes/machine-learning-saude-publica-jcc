document.addEventListener('DOMContentLoaded', () => {
    const mainStages = document.querySelectorAll('.main-stage-box');
    
    mainStages.forEach(stage => {
        const targetId = stage.getAttribute('data-target');
        const popup = document.getElementById(targetId);

        if (!popup) return;
        
        const positionPopup = () => {
            const stageRect = stage.getBoundingClientRect();
            const popupRect = popup.getBoundingClientRect();
            let left = stageRect.left + (stageRect.width / 2) - (popupRect.width / 2);
            
            if (left < 10) left = 10;
            if (left + popupRect.width > window.innerWidth - 10) {
                left = window.innerWidth - popupRect.width - 10;
            }

            popup.style.left = `${left}px`;
            popup.style.top = `${stageRect.bottom + window.scrollY + 15}px`;
        };

        stage.addEventListener('mouseenter', () => {
            document.querySelectorAll('.details-popup').forEach(p => p.classList.remove('visible'));
            popup.classList.add('visible');
            positionPopup();
        });

        popup.addEventListener('mouseenter', () => {
            popup.classList.add('visible');
        });

        stage.addEventListener('mouseleave', () => {
            setTimeout(() => {
                if (!popup.matches(':hover')) {
                    popup.classList.remove('visible');
                }
            }, 200);
        });

        popup.addEventListener('mouseleave', () => {
            popup.classList.remove('visible');
        });
    });
    
    window.addEventListener('resize', () => {
        document.querySelectorAll('.details-popup.visible').forEach(popup => {
            const targetId = popup.id;
            const stage = document.querySelector(`[data-target="${targetId}"]`);
            if(stage) {
                    const stageRect = stage.getBoundingClientRect();
                    const popupRect = popup.getBoundingClientRect();
                    let left = stageRect.left + (stageRect.width / 2) - (popupRect.width / 2);
                    if (left < 10) left = 10;
                    if (left + popupRect.width > window.innerWidth - 10) {
                        left = window.innerWidth - popupRect.width - 10;
                    }
                    popup.style.left = `${left}px`;
                    popup.style.top = `${stageRect.bottom + window.scrollY + 15}px`;
            }
        });
    });
});