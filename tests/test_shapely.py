from shapely import MultiPolygon, Polygon, intersection, union_all

ob1 = MultiPolygon([(
    
    ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),

    [((0.1,0.1), (0.1,0.2), (0.2,0.2), (0.2,0.1))]

    )

])


if __name__ == '__main__':
    obj = Polygon(((1, 1), (2, 2), (3, 1), (2, 0)))
    rect = Polygon(((0, 0), (2, 0), (2, 2), (0, 2)))
    print(rect.intersects(obj))
    inters = intersection(obj, rect)
    print(inters)
    print(inters.exterior.coords.xy[0].tolist())
    
    union_poly = union_all([
            Polygon([[0, 0], [0, 1], [1, 0]]), 
            Polygon([[0.1, 0.2], [0, 0.5], [0.5, 0]])])
    print(
        union_poly.equals(Polygon([[0, 0], [0, 1], [1, 0]]))
    )
    