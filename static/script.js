let mobileForm = document.getElementById("mobile-form")
let resnetForm = document.getElementById("resnet-form")
let mobileButton = document.getElementById("mobileButton")
let resnetButton = document.getElementById("resnetButton")


function mobile() {
    mobileButton.classList.remove("btn-outline-dark");
    mobileButton.classList.add("btn-dark");
    resnetButton.classList.remove("btn-dark");
    resnetButton.classList.add("btn-outline-dark");
    resnetForm.style.opacity = "0";
    setTimeout(() => {
        resnetForm.style.display = "none";
        mobileForm.style.opacity = "1";
    }, 300); 
    setTimeout(() => {
        mobileForm.style.display = "block";
    }, 500);

};
function resnet() {
    mobileButton.classList.remove("btn-dark");
    mobileButton.classList.add("btn-outline-dark");
    resnetButton.classList.remove("btn-outline-dark");
    resnetButton.classList.add("btn-dark");
    mobileForm.style.opacity = "0";
    setTimeout(() => {
        mobileForm.style.display = "none";
        resnetForm.style.opacity = "1";
    }, 300); 
    setTimeout(() => {
        resnetForm.style.display = "block";
    }, 500);
};

let rate = document.getElementById("rate");
let correct = document.getElementById("correct");
let incorrect = document.getElementById("incorrect");

rate.addEventListener("click", function () {
    document.body.classList.toggle('user-feed');
  if (incorrect.style.visibility === "hidden") {
    incorrect.style.visibility = "visible";
    correct.style.visibility = "visible";
    
    
    setTimeout(() => {
        incorrect.style.opacity = "1"; 
        correct.style.opacity = "1";
    }, 300); 

    rate.textContent = "Cancel";
    rate.style.backgroundColor = "#DC3545";
    
  } else {
    incorrect.style.opacity = "0"; 
    correct.style.opacity = "0";

    setTimeout(() => {
      incorrect.style.visibility = "hidden"; 
      correct.style.visibility = "hidden";
    }, 20); 
    
    rate.textContent = "Feedback";
    rate.style.backgroundColor = "#343A40";
  }
});



function readURL1(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('#MobileNetv2image').attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
    }
}
function readURL2(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('#ResNet50image').attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
    }
}
function runOnRefreshOrLoad() {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('#image').attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
    }
}
window.onload = runOnRefreshOrLoad;