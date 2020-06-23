function [p, e, t] = meshfrompolyverts(X, hmax)
  pg = polyshape(X(:, 1), X(:, 2))
  tr = triangulation(pg);
  model = createpde();
  geom = geometryFromMesh(model, tr.Points', tr.ConnectivityList');
  femmesh = generateMesh(model, 'Hmax', hmax, 'GeometricOrder', 'linear');
  [p, e, t] = meshToPet(femmesh);
