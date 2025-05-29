$(document).on('click', '#generate_image', function() {
    let image_size = $('#select_image_size').find(':selected').text();
    let seed = $('#input_seed').val();
    let color = $('#select_color').find(':selected').val();
    if(image_size === '24x40') {
        data = 'http://localhost:5000/image_color?res=' + image_size + '&target_color=' + color + '&seed=' + seed;
    } else {
        data = 'http://localhost:5000/image?res=' + image_size + '&timestamp=' + new Date().getTime()
    }
    
    $('.generated_image').attr('src', data);
});

$(document).on('mouseup', '#interpolation_range', function() {
    let seed_left = $('#input_seed_left').val();
    let seed_right = $('#input_seed_right').val();
    let interpolation_range = $('#interpolation_range').val();
    let timestamp = '&timestamp=' + new Date().getTime();
    let image_size = $('#interpolation_image_size').find(':selected').text();
    middle_image = 'http://localhost:5000/interpolate?res=' + image_size + '&seed1=' + seed_left + '&seed2=' + seed_right + '&t=' + interpolation_range + timestamp;
    $('.image-center').attr('src', middle_image);
});

$(document).on('click', '#left_image_btn', function() {
    let seed_left = $('#input_seed_left').val();
    let timestamp = '&timestamp=' + new Date().getTime();
    let image_size = $('#interpolation_image_size').find(':selected').text();
    left_image = 'http://localhost:5000/image?res=' + image_size + '&seed=' + seed_left + timestamp;
    $('.image-left').attr('src', left_image);
});

$(document).on('click', '#right_image_btn', function() {
    let seed_right = $('#input_seed_right').val();
    let timestamp = '&timestamp=' + new Date().getTime();
    let image_size = $('#interpolation_image_size').find(':selected').text();
    right_image = 'http://localhost:5000/image?res=' + image_size + '&seed=' + seed_right + timestamp;
    $('.image-right').attr('src', right_image);
});

$(document).on('change', '#interpolation_image_size', function() {
    let seed_left = $('#input_seed_left').val();
    let seed_right = $('#input_seed_right').val();
    let interpolation_range = $('#interpolation_range').val();
    let timestamp = '&timestamp=' + new Date().getTime();
    let image_size = $('#interpolation_image_size').find(':selected').text();
    left_image = 'http://localhost:5000/image?res=' + image_size + '&seed=' + seed_left + timestamp;
    right_image = 'http://localhost:5000/image?res=' + image_size + '&seed=' + seed_right + timestamp;
    middle_image = 'http://localhost:5000/interpolate?res=' + image_size + '&seed1=' + seed_left + '&seed2=' + seed_right + '&t=' + interpolation_range + timestamp;
    $('.image-center').attr('src', middle_image);
    $('.image-left').attr('src', left_image);
    $('.image-right').attr('src', right_image);
})