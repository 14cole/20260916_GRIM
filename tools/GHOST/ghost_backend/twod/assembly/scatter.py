"""Route Galerkin contributions straight into the owned equation matrix."""
import numpy as np
from ghost_backend.twod.assembly.compact import CompactOperator


class SystemScatter:
    def __init__(self, matrix, node_count, rows, columns, routes):
        self.matrix = matrix
        self.row_ids = CompactOperator._ids(rows, node_count)
        self.column_ids = CompactOperator._ids(columns, node_count)
        self.node_count = node_count
        self._maps = None
        self.routes = routes

    def _node_maps(self):
        # Built on first use: compressed tiles create many destinations whose
        # routes carry their own maps and never read these.
        if self._maps is None:
            row_map = np.full(self.node_count, -1, dtype=np.int64)
            column_map = np.full(self.node_count, -1, dtype=np.int64)
            row_map[self.row_ids] = np.arange(len(self.row_ids))
            column_map[self.column_ids] = np.arange(len(self.column_ids))
            self._maps = row_map, column_map
        return self._maps

    @property
    def row_map(self):
        return self._node_maps()[0]

    @property
    def column_map(self):
        return self._node_maps()[1]

    def scatter_add(self, rows, columns, values):
        rows, columns = np.asarray(rows), np.asarray(columns)
        if rows.ndim == 2 and columns.ndim == 2 and rows.shape[1] == 1 and columns.shape[0] == 1:
            self._scatter_outer(rows[:, 0], columns[0], values)
            return
        for row_map, column_map, weights in self.routes:
            rr, cc = np.broadcast_arrays(row_map[rows], column_map[columns])
            keep = (rr >= 0) & (cc >= 0)
            if np.any(keep):
                scaled = np.broadcast_to(values, keep.shape) * np.broadcast_to(weights[rows], keep.shape)
                np.add.at(self.matrix, (rr[keep], cc[keep]), scaled[keep])

    def scatter_add_columns(self, rows, columns, values):
        """scatter_add(rows[:, None], columns[None, :, b], values[b]) for each b in order.

        Distinct routes of one destination never write the same entry (they differ
        in equation rows or in source-side columns), so one pass per route keeps
        every entry's sum order.
        """
        rows, columns = np.asarray(rows), np.asarray(columns)
        width = columns.shape[1]
        values = np.broadcast_to(values, (width, len(rows), len(columns)))
        if type(self).scatter_add is not SystemScatter.scatter_add:
            for b in range(width):
                self.scatter_add(rows[:, None], columns[None, :, b], values[b])
            return
        matrix = self.matrix
        from ghost_backend.twod.assembly.native.far import scatter_columns
        for row_map, column_map, weights in self.routes:
            if scatter_columns(matrix, rows, columns, row_map, column_map, values * weights[rows][None, :, None]):
                continue
            ri = np.flatnonzero(row_map[rows] >= 0)
            if not len(ri):
                continue
            cc = column_map[columns].T
            keep = cc >= 0
            if not keep.any():
                continue
            r = row_map[rows[ri]]
            if len(ri) == len(rows):
                block = values * weights[rows][None, :, None]
            else:
                block = values[:, ri, :] * weights[rows[ri]][None, :, None]
            keep = np.broadcast_to(keep[:, None, :], block.shape)
            if matrix.flags.f_contiguous:
                index = r[None, :, None] + cc[:, None, :] * matrix.shape[0]
                np.add.at(matrix.reshape(-1, order='F'), index[keep], block[keep])
            elif matrix.flags.c_contiguous:
                index = r[None, :, None] * matrix.shape[1] + cc[:, None, :]
                np.add.at(matrix.reshape(-1), index[keep], block[keep])
            else:
                rr = np.broadcast_to(r[None, :, None], block.shape)
                np.add.at(matrix, (rr[keep], np.broadcast_to(cc[:, None, :], block.shape)[keep]), block[keep])

    def _scatter_outer(self, rows, columns, values):
        """rows x columns tiles: same sums as scatter_add, without per-entry index arrays."""
        values = np.broadcast_to(values, (len(rows), len(columns)))
        for row_map, column_map, weights in self.routes:
            ri = np.flatnonzero(row_map[rows] >= 0)
            if not len(ri):
                continue
            ci = np.flatnonzero(column_map[columns] >= 0)
            if not len(ci):
                continue
            r, c = row_map[rows[ri]], column_map[columns[ci]]
            if len(ri) == len(rows) and len(ci) == len(columns):
                block = values * weights[rows][:, None]
            else:
                block = values[np.ix_(ri, ci)] * weights[rows[ri]][:, None]
            matrix = self.matrix
            # One-dimensional ufunc.at over a contiguous view is several times
            # faster than 2-D fancy indexing and adds duplicates in the same order.
            if matrix.flags.f_contiguous:
                np.add.at(matrix.reshape(-1, order='F'), (r[:, None] + c[None, :] * matrix.shape[0]).ravel(), block.ravel())
            elif matrix.flags.c_contiguous:
                np.add.at(matrix.reshape(-1), (r[:, None] * matrix.shape[1] + c[None, :]).ravel(), block.ravel())
            else:
                np.add.at(matrix, (r[:, None], c[None, :]), block)


class MatrixDestination(SystemScatter):
    """Accumulate into an owned full operator or a non-contiguous A block."""
    def __init__(self, matrix, node_count):
        if not isinstance(matrix, np.ndarray) or matrix.shape != (node_count, node_count):
            raise ValueError('Operator destination must match the mesh nodes.')
        if matrix.dtype != np.complex128 or not matrix.flags.writeable:
            raise ValueError('Operator destination must be writable complex128 storage.')
        ids = np.arange(node_count)
        super().__init__(matrix, node_count, ids, ids, [])

    def scatter_add(self, rows, columns, values):
        np.add.at(self.matrix, (rows, columns), values)


def robin_outputs(matrix, mesh, requests):
    n = len(mesh.nodes)
    columns, weights = np.arange(n), np.ones(n, complex)
    outputs = []
    for srows, krows, coefficients in requests:

        if coefficients is None:
            matrix[np.asarray(srows, int)] = 0
        pair = []
        for rows in (srows, krows):
            row_map = np.full(n, -1, int)
            row_map[rows] = rows
            routes = [(row_map, columns, weights)] if len(rows) else []
            pair.append(SystemScatter(matrix, n, rows, columns, routes))
        outputs.append(tuple(pair))
    return outputs


def multi_outputs(matrix, mesh, layout, k, requests):
    """One kernel request may feed distinct media with identical wavenumber."""
    from ghost_backend.twod.formulations.regions import _inverse_beta
    from ghost_backend.twod.constants import EPS
    n = len(mesh.nodes)
    ifaces, dofs, regions = layout['ifaces'], layout['dof_map'], layout['region_props']
    results = []
    for request in requests:
        source = request['source']
        iface = ifaces[source]
        s_routes, k_routes = [], []
        for rid, mis in layout['region_ifaces'].items():
            if regions[rid]['k'] != k or source not in mis:
                continue
            col = np.full(n, -1, np.int64)
            offset, count = dofs[source, 'minus' if iface['r_m'] == rid else 'plus']
            col[iface['nodes']] = np.arange(offset, offset+count)
            sr, kr = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
            sw, kw = np.zeros(n, complex), np.zeros(n, complex)
            for observer in mis:
                target = ifaces[observer]
                nodes = np.asarray(target['nodes'])
                rm, rp = target['r_m'], target['r_p']
                if request['kind'] == 'weighted' and observer != request['observer']:
                    continue
                if rm < 0 or rp < 0:
                    offset, count = dofs[observer, 'plus' if rm < 0 else 'minus']
                    dest = np.arange(offset, offset+count)
                    pec = np.abs(target['robin_alpha']) <= EPS if layout['polarization'] == 'TM' else np.zeros(count, bool)
                    if request['kind'] == 'weighted':
                        sr[nodes[~pec]], sw[nodes[~pec]] = dest[~pec], 1
                    else:
                        sr[nodes[pec]], sw[nodes[pec]] = dest[pec], 1
                        kr[nodes[~pec]], kw[nodes[~pec]] = dest[~pec], 1
                elif request['kind'] == 'plain':
                    flux, count = dofs[observer, 'minus']
                    trace, _ = dofs[observer, 'plus']
                    sr[nodes], kr[nodes] = np.arange(trace, trace+count), np.arange(flux, flux+count)
                    sw[nodes] = 1 if rid == rm else -1
                    kw[nodes] = 1 if rid == rm else -_inverse_beta(layout, target, layout['polarization'])
            if np.any(sr >= 0):
                s_routes.append((sr, col, sw))
            if np.any(kr >= 0):
                k_routes.append((kr, col, kw))
        results.append((SystemScatter(matrix, n, sorted(request['s_rows']), iface['nodes'], s_routes),
                        SystemScatter(matrix, n, sorted(request['k_rows']), iface['nodes'], k_routes)))
    return results


def assemble_multi(mesh, infos, pol, obs_order, src_order):
    import ghost_backend.twod.solver as rcs
    from ghost_backend.twod.formulations.regions import (
        build_layout,
        operator_plan,
        _sparse_mass,
        _inverse_beta,
    )
    layout = build_layout(mesh, infos, pol)
    matrix = np.zeros((layout['n_dof'], layout['n_dof']), complex, order='F')
    mass = _sparse_mass(mesh)
    for mi, interface in enumerate(layout['ifaces']):
        nodes = interface['nodes']
        local = mass[nodes, :][:, nodes].tocoo()
        rm, rp = interface['r_m'], interface['r_p']
        if rm < 0 or rp < 0:
            offset, _ = layout['dof_map'][mi, 'plus' if rm < 0 else 'minus']
            keep = np.ones(local.nnz, bool)
            if pol == 'TM':
                keep = np.abs(interface['robin_alpha'][local.row]) > rcs.EPS
            matrix[offset+local.row[keep], offset+local.col[keep]] += (.5 if rm < 0 else -.5)*local.data[keep]
        else:
            flux, _ = layout['dof_map'][mi, 'minus']
            trace, _ = layout['dof_map'][mi, 'plus']
            matrix[flux+local.row, flux+local.col] -= .5*local.data
            matrix[flux+local.row, trace+local.col] -= .5*_inverse_beta(layout, interface, pol)*local.data
    mass = None
    for k, requests in operator_plan(layout):
        ifaces = layout['ifaces']
        rcs._assemble_linear_operator_matrices_multi(mesh, k, True,
            [ifaces[r['source']]['mask'] for r in requests], obs_order=obs_order, src_order=src_order,
            compute_double_layer_many=[r['kind'] == 'plain' for r in requests],
            single_layer_observation_coefficients_many=[None if r['observer'] is None else
                ifaces[r['observer']]['robin_alpha_elements'] for r in requests],
            output_node_ids_many=[(sorted(r['s_rows']), ifaces[r['source']]['nodes']) for r in requests],
            double_layer_output_node_ids_many=[(sorted(r['k_rows']), ifaces[r['source']]['nodes']) for r in requests],
            operator_outputs=multi_outputs(matrix, mesh, layout, k, requests))
    return matrix, layout
