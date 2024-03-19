const leftarea = ["Onboarding","Dashboard","Orders","Products","Customers","Promotions","Pages","Languages","Themes","Sales Channels","Apps"];

const getingleftbottomarea = document.getElementsByClassName("bottom-left-second-main-page")[0];
for(let i=0; i<leftarea.length; ++i)
{
    const newDiv = document.createElement("div");
    newDiv.className = "data-bottom-left-second-main-page";
    const newAnchor = document.createElement("a");
    newAnchor.href = '#';
    newAnchor.textContent = leftarea[i];
    newDiv.appendChild(newAnchor);
    getingleftbottomarea.appendChild(newDiv);
}

function displayImage(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            var imagePreview = document.getElementById('image-preview');
            imagePreview.innerHTML = '<img src="' + e.target.result + '" width="300" height="300">';
        };

        reader.readAsDataURL(input.files[0]);
    }
}
