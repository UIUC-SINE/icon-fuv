from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# create new figure, axes instances.
fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = Basemap(llcrnrlon=-180.,llcrnrlat=-30.,urcrnrlon=180.,urcrnrlat=30.,\
            resolution='l',projection='merc',\
            # lat_0=40.,lon_0=-20.,lat_ts=20.
)
m.drawcoastlines()
m.fillcontinents()
m.drawparallels(np.arange(-30,30,20),labels=[1,1,0,1])
# draw meridians
m.drawmeridians(np.arange(-180,180,60),labels=[1,1,0,1])
date = datetime.utcnow()
m.nightshade(date)
ax.set_title('Great Circle from New York to London')
plt.show()
