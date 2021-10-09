// Owl slider
$(document).ready(function () {
  $(".owl-carousel").owlCarousel({
    loop: true,
    margin: 20,
    nav: true,
    navText: ["<i class='fa fa-angle-left'></i>", "<i class='fa fa-angle-right'></i>"],
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

// if sample is selected, set submit disable to false
$(document).on("click", "input[name='sample']", function () {
  $("input:submit").prop("disabled", false);
});

// when file is uploaded, check size; if bigger than 50MB, set alert;
// otherwise, set submit disable to false
$(document).on("change", "input:file", function () {
  var file = this.files[0];
  if (file.size > 104857600) {
    alert("Video must be smaller than 100MB.");
    $(this).replaceWith('<input type="file" id="file" name="upload" class="file" />');
    $(".file-name").replaceWith('<p class="file-name"></p>');
    return false;
  } else {
    $("input:submit").prop("disabled", false);
    var size = (this.files[0].size / 1024 / 1024).toFixed(2);
    var name = this.files[0].name;
    var fileNameAndSize = `${name} - ${size}MB`;
    document.querySelector(".file-name").textContent = fileNameAndSize;
  }
});

$(document).on("click", "#submit", function () {
  $("#non_loading").hide();
  $("#loading").show();
});
