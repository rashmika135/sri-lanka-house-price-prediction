const form = document.getElementById("price-form");
const predictButton = document.getElementById("predict-button");
const buttonText = document.getElementById("button-text");
const errorMessage = document.getElementById("error-message");
const postedYear = document.getElementById("posted_year");

postedYear.max = new Date().getFullYear();

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const house = {
        bedrooms: Number(form.bedrooms.value),
        bathrooms: Number(form.bathrooms.value),
        house_size: Number(form.house_size.value),
        land_size_perches: Number(form.land_size_perches.value),
        location: form.location.value.trim(),
        exact_area: form.exact_area.value,
        geo_region: form.geo_region.value,
        membership_level: form.membership_level.value,
        is_verified: form.is_verified.checked,
        is_member: form.is_member.checked,
        is_authorized_dealer: form.is_authorized_dealer.checked,
        posted_year: Number(form.posted_year.value),
    };

    errorMessage.hidden = true;
    predictButton.disabled = true;
    buttonText.textContent = "Calculating estimate...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(house),
        });
        const result = await response.json();

        if (!response.ok) {
            const detail = result.detail;
            const message = Array.isArray(detail)
                ? detail.map((item) => item.msg).join(". ")
                : detail || "Could not calculate the price.";
            throw new Error(message);
        }

        document.getElementById("predicted-price").textContent =
            "Rs " + new Intl.NumberFormat("en-US").format(result.predicted_price_lkr);
        document.getElementById("predicted-millions").textContent =
            "Rs " + result.predicted_price_millions.toFixed(2) + " million";

        document.getElementById("empty-result").hidden = true;
        document.getElementById("filled-result").hidden = false;

        if (window.innerWidth < 880) {
            document.getElementById("result-card").scrollIntoView({ behavior: "smooth" });
        }
    } catch (error) {
        errorMessage.textContent = error.message || "Unable to connect to the API.";
        errorMessage.hidden = false;
    } finally {
        predictButton.disabled = false;
        buttonText.textContent = "Estimate house price";
    }
});
