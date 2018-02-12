package it.unipd.dei.dm1617.project;

import scala.Serializable;

public class KCenterPoint implements Comparable, Serializable {
    VectorWikiPage page;
    long idCenter;
    double distanceFromCenter;
    double totalDistance;

    KCenterPoint(VectorWikiPage page, long idCenter, double distanceFromCenter, double totalDistance) {
        this.page = page;
        this.idCenter = idCenter;
        this.distanceFromCenter = distanceFromCenter;
        this.totalDistance = totalDistance;
    }

    @Override
    public int compareTo(Object o) {
        return (int) (this.totalDistance - ((KCenterPoint) o).totalDistance);
    }
}