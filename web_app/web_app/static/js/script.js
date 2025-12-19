// Scholarship Recommender System - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Handle feedback form submissions
    document.querySelectorAll('.feedback-form').forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const studentId = this.dataset.student;
            const scholarshipId = this.dataset.scholarship;
            
            // Add additional data
            formData.append('student_id', studentId);
            formData.append('scholarship_id', scholarshipId);
            
            submitFeedback(formData, this);
        });
    });

    // Auto-collapse other feedback forms when one is opened
    document.querySelectorAll('.feedback-btn').forEach(button => {
        button.addEventListener('click', function() {
            const target = this.dataset.bsTarget;
            
            // Close all other open feedback forms
            document.querySelectorAll('.collapse.show').forEach(openCollapse => {
                if (openCollapse.id !== target.substring(1)) {
                    bootstrap.Collapse.getInstance(openCollapse).hide();
                }
            });
        });
    });

    // Add loading animation for form submissions
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                submitBtn.disabled = true;
            }
        });
    });
});

function submitFeedback(formData, formElement) {
    fetch('/feedback', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showAlert('Thank you for your feedback!', 'success');
            
            // Reset form and collapse
            formElement.reset();
            const collapseElement = formElement.closest('.collapse');
            if (collapseElement) {
                bootstrap.Collapse.getInstance(collapseElement).hide();
            }
        } else {
            showAlert('Error submitting feedback: ' + data.error, 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error submitting feedback. Please try again.', 'danger');
    })
    .finally(() => {
        // Re-enable submit button
        const submitBtn = formElement.querySelector('button[type="submit"]');
        if (submitBtn) {
            submitBtn.innerHTML = '<i class="fas fa-check"></i> Submit';
            submitBtn.disabled = false;
        }
    });
}

function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.querySelector('.container').insertBefore(alertDiv, document.querySelector('.container').firstChild);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// API functions for future expansion
class ScholarshipAPI {
    static async getRecommendations(studentId, topN = 10) {
        try {
            const response = await fetch(`/api/recommend/${studentId}?top_n=${topN}`);
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            return null;
        }
    }
    
    static async searchScholarships(query) {
        // This would connect to a search endpoint when implemented
        console.log('Searching for:', query);
        return [];
    }
}

// Utility functions
function formatScore(score) {
    return (score * 100).toFixed(1) + '%';
}

function getScoreColor(score) {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'secondary';
}