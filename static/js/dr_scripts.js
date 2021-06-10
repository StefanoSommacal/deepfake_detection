// Owl slider
$(document).ready(function () {
  $(".owl-carousel").owlCarousel({
    loop: true,
    margin: 20,
    nav: true,
    navText: [
      "<i class='fa fa-angle-left'></i>",
      "<i class='fa fa-angle-right'></i>",
    ],
    responsiveClass: true,
    responsive: {
      0: {
        items: 1,
        nav: true,
      },
      600: {
        items: 3,
        nav: false,
      },
      1000: {
        items: 3,
        nav: true,
        loop: false,
      },
    },
  });
});

$(".owl-nav").css({
  display: "none",
  visibility: "hidden",
});

$(".owl-nav").hide();

// if model is selected and video is uploaded, set button disable false
$(document).on("change", "input:file", function () {
  if ($("input:file").val()) {
    $("input:submit").prop("disabled", false);
  }
});

// if model is selected and sample is selected, set button disable false
$(document).on("click", "input[name='sample']", function () {
  $("input:submit").prop("disabled", false);
});

// if video or sample and then model, set button disable false
$(document).on("change", "#model_selection", function () {
  if ($(this).val() == "---") {
    $("input:submit").prop("disabled", true);
  }
  if ($(this).val() != "---" && $("input:file").val()) {
    $("input:submit").prop("disabled", false);
  }
  if ($(this).val() != "---" && $("input[name='sample']").is(":checked")) {
    $("input:submit").prop("disabled", false);
  }
});

// when video is uploaded, display name and size
$(document).on("change", "#file", function () {
  var size = (this.files[0].size / 1024 / 1024).toFixed(2);
  var name = this.files[0].name;
  var fileNameAndSize = `${name} - ${size}MB`;
  document.querySelector(".file-name").textContent = fileNameAndSize;
});

$(document).on("click", "#submit", function () {
  $("#non_loading").hide();
  $("#loading").show();
});
