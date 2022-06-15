# We need selenium to automate our process
from selenium import webdriver
# below import helps search some elements with By.ID or similar expressions.
from selenium.webdriver.common.by import By

# we need basemap for draw_map()
from mpl_toolkits.basemap import Basemap
import numpy as np
import random
import time


# below function is a simple Basemap module code snippet. it draws a map of Earth with only lands as valid values.
def draw_map():
    bottomlat = -89.0
    toplat = 89.0
    bottomlong = -170.0
    toplong = 170.0
    world_map = Basemap(projection="merc", resolution='c', area_thresh=0.1, llcrnrlon=bottomlong, llcrnrlat=bottomlat, urcrnrlon=toplong, urcrnrlat=toplat)
    world_map.drawcoastlines(color='black')
    return world_map


def main():
    # Below are firefox profile settings
    # deprecation warnings are just warnings. the code works regardless.
    profile = webdriver.FirefoxProfile()
    profile.set_preference('browser.download.manager.showWhenStarting', False)
    profile.set_preference('browser.download.folderList', 2)
    profile.set_preference('browser.download.dir', '/tmp')
    profile.set_preference("browser.download.dir", "C:\\heightmaps\\new_heightmaps\\")# intended directory
    profile.set_preference("browser.download.useDownloadDir", True)
    # below 2 attributes are very crucial. they help allow skip the downloading pop-up dialogue of firefox browser.
    profile.set_preference('browser.download.improvements_to_download_panel', True)
    profile.set_preference('browser.helperApps.neverAsk.saveToDisk', "image/png")

    # initiate the firefox driver with our profile modified above
    driver = webdriver.Firefox(profile)

    # we get our heightmap data from the url below.
    driver.get("https://heightmap.skydark.pl")

    # these driver buttons are hand tailored for "https://heightmap.skydark.pl".
    # would need modifying for any other heightmap source. Check related xpath documentation of Selenium webdriver.
    search_button = driver.find_element_by_xpath("//a[@title='Set lng/lat']")
    save_button = driver.find_element_by_xpath("//a[@title='Download PNG height map']")
    lng_field = driver.find_element(By.ID, "lngInput")
    lat_field = driver.find_element(By.ID, "latInput")
    apply_search_button = driver.find_element_by_xpath("//button[@onclick='setLngLat(2)']")

    # draw the map to use in 'while True' to check for lands.
    world_map = draw_map()

    # this sleeps are to make sure that the website is loaded properly before we start looping our requests.
    time.sleep(12)
    search_button.click()
    time.sleep(3)

    # i = number of heightmap.png files to search and download.
    # we have utilized values like 100, 500, 1000 etc. Any value is ok.
    # just a warning, sometimes the site has a problem with how often our program queries, and it jams the program.
    # It is our advice that whoever is using this, should sometimes check if it is still actively downloading png files.
    for i in range(550):
        while True:
            # generate a random longitude and latitude value
            lon, lat = random.uniform(-179, 179), random.uniform(-89, 89)

            # convert to projection map for checking purposes in the 'if' below
            xpt, ypt = world_map(lon, lat)

            # Check if that point is on the land, we need land heightmaps to have height deviation
            if world_map.is_land(xpt, ypt):
                # if it is on the land break our 'while True' and go to our 'try:'
                break

        try:
            # clear lng and lat fields to input fresh values later
            lng_field.clear()
            lat_field.clear()

            # input fresh lng and lat values to the search tool
            lng_field.send_keys(lon)
            lat_field.send_keys(lat)

            # search the intended land with the inputted lng and lat values
            apply_search_button.click()

            # wait for the searched area to load, otherwise we cannot get consistent returns
            time.sleep(2)

            # save the heightmap.png to our directory selected at the beginning
            save_button.click()
            # time to sleep, which helps us to have no problem with our downloads
            time.sleep(1)
        except:
            input("loop is thrown out with exception")


if __name__ == "__main__":
    main()
