/* Optional native kernel for vpxc.
 *
 * The Python modules remain the bit-exact reference. This file provides:
 *   - a 65536-entry Core66 symbol lookup (no bsearch in the decode inner loop)
 *   - 32-lane interleaved rANS put/get that follows src/vpxc/interleaved.py
 *   - a 1-D Q15.16 sequential I-SSM scan matching vpxc.issm.sequential
 *
 * Build (from the repo root):
 *   cc -O3 -fPIC -shared $(python3 -c "import sysconfig; print('-I' + sysconfig.get_path('include'))") \
 *      csrc/vpxc_accel.c -o src/vpxc/_accel$(python3-config --extension-suffix)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define RANS64_L ((uint64_t)1 << 31)
#define RANS_M 65536u
#define LANES 32

static int build_lut(const int32_t *cum, int nsym, uint8_t *lut) {
    if (nsym < 1 || nsym > 255) return -1;
    for (int sym = 0; sym < nsym; ++sym) {
        int start = cum[sym];
        int stop = cum[sym + 1];
        if (start < 0 || stop > (int)RANS_M || stop < start) return -1;
        for (int cf = start; cf < stop; ++cf) lut[cf] = (uint8_t)sym;
    }
    return 0;
}

static PyObject *accel_symbol_lut(PyObject *self, PyObject *args) {
    Py_buffer cum;
    if (!PyArg_ParseTuple(args, "y*", &cum)) return NULL;
    if (cum.len < (Py_ssize_t)(2 * (Py_ssize_t)sizeof(int32_t)) ||
        (cum.len % (Py_ssize_t)sizeof(int32_t)) != 0) {
        PyBuffer_Release(&cum);
        PyErr_SetString(PyExc_ValueError, "cumulative buffer is not int32");
        return NULL;
    }
    int nsym = (int)(cum.len / (Py_ssize_t)sizeof(int32_t)) - 1;
    uint8_t *lut = (uint8_t *)malloc(RANS_M);
    if (!lut) {
        PyBuffer_Release(&cum);
        return PyErr_NoMemory();
    }
    if (build_lut((const int32_t *)cum.buf, nsym, lut) != 0) {
        free(lut);
        PyBuffer_Release(&cum);
        PyErr_SetString(PyExc_ValueError, "cumulative table is not ordered");
        return NULL;
    }
    PyObject *out = PyBytes_FromStringAndSize((const char *)lut, RANS_M);
    free(lut);
    PyBuffer_Release(&cum);
    return out;
}

/* One lane mirrors RansEncoder.put / finish. */
typedef struct {
    uint64_t state;
    uint32_t *words;
    int nwords;
    int cap;
} Lane;

static int lane_push_word(Lane *lane, uint32_t word) {
    if (lane->nwords >= lane->cap) {
        int cap = lane->cap ? lane->cap * 2 : 16;
        uint32_t *grown = (uint32_t *)realloc(lane->words, (size_t)cap * sizeof(uint32_t));
        if (!grown) return -1;
        lane->words = grown;
        lane->cap = cap;
    }
    lane->words[lane->nwords++] = word;
    return 0;
}

static int lane_put(Lane *lane, uint32_t start, uint32_t freq) {
    if (freq == 0) return -1;
    uint64_t x = lane->state;
    uint64_t x_max = ((RANS64_L >> 16) << 32) * (uint64_t)freq;
    while (x >= x_max) {
        if (lane_push_word(lane, (uint32_t)(x & 0xffffffffu)) != 0) return -1;
        x >>= 32;
    }
    uint64_t q = x / freq;
    uint64_t r = x % freq;
    lane->state = (q << 16) + (uint64_t)start + r;
    return 0;
}

static PyObject *lane_bytes(Lane *lane) {
    /* finish(): append lo, hi; emit hi, lo, then renorm words reversed. */
    uint32_t lo = (uint32_t)(lane->state & 0xffffffffu);
    uint32_t hi = (uint32_t)((lane->state >> 32) & 0xffffffffu);
    if (lane_push_word(lane, lo) != 0 || lane_push_word(lane, hi) != 0) return NULL;
    PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)lane->nwords * 4);
    if (!out) return NULL;
    char *buf = PyBytes_AS_STRING(out);
    /* words[-1] is hi, words[-2] is lo, then reversed prefix. */
    uint32_t *dst = (uint32_t *)buf;
    dst[0] = lane->words[lane->nwords - 1];
    dst[1] = lane->words[lane->nwords - 2];
    for (int i = 0; i < lane->nwords - 2; ++i) {
        dst[2 + i] = lane->words[lane->nwords - 3 - i];
    }
    return out;
}

static PyObject *accel_interleaved_encode(PyObject *self, PyObject *args) {
    PyObject *pairs;
    if (!PyArg_ParseTuple(args, "O", &pairs)) return NULL;
    Py_ssize_t n = PySequence_Size(pairs);
    if (n < 0) return NULL;
    Lane lanes[LANES];
    memset(lanes, 0, sizeof(lanes));
    for (int i = 0; i < LANES; ++i) lanes[i].state = RANS64_L;
    for (Py_ssize_t index = n - 1; index >= 0; --index) {
        PyObject *item = PySequence_GetItem(pairs, index);
        if (!item) goto fail;
        int start, freq;
        if (!PyArg_ParseTuple(item, "ii", &start, &freq)) {
            Py_DECREF(item);
            goto fail;
        }
        Py_DECREF(item);
        if (lane_put(&lanes[index % LANES], (uint32_t)start, (uint32_t)freq) != 0) {
            PyErr_SetString(PyExc_RuntimeError, "interleaved put failed");
            goto fail;
        }
    }
    PyObject *blobs[LANES];
    memset(blobs, 0, sizeof(blobs));
    Py_ssize_t total = 0;
    for (int i = 0; i < LANES; ++i) {
        blobs[i] = lane_bytes(&lanes[i]);
        if (!blobs[i]) goto fail;
        total += 4 + PyBytes_GET_SIZE(blobs[i]);
    }
    PyObject *out = PyBytes_FromStringAndSize(NULL, total);
    if (!out) goto fail;
    char *dst = PyBytes_AS_STRING(out);
    Py_ssize_t pos = 0;
    for (int i = 0; i < LANES; ++i) {
        uint32_t len = (uint32_t)PyBytes_GET_SIZE(blobs[i]);
        memcpy(dst + pos, &len, 4);
        pos += 4;
        memcpy(dst + pos, PyBytes_AS_STRING(blobs[i]), len);
        pos += len;
        Py_DECREF(blobs[i]);
        blobs[i] = NULL;
        free(lanes[i].words);
        lanes[i].words = NULL;
    }
    return out;
fail:
    for (int i = 0; i < LANES; ++i) {
        Py_XDECREF(blobs[i]);
        free(lanes[i].words);
    }
    return NULL;
}

static PyObject *accel_sequential_scan(PyObject *self, PyObject *args) {
    Py_buffer decay, bias;
    if (!PyArg_ParseTuple(args, "y*y*", &decay, &bias)) return NULL;
    if (decay.len != bias.len || (decay.len % (Py_ssize_t)sizeof(int64_t)) != 0) {
        PyBuffer_Release(&decay);
        PyBuffer_Release(&bias);
        PyErr_SetString(PyExc_ValueError, "decay and bias must be matching int64 buffers");
        return NULL;
    }
    Py_ssize_t n = decay.len / (Py_ssize_t)sizeof(int64_t);
    int64_t *out = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    if (!out) {
        PyBuffer_Release(&decay);
        PyBuffer_Release(&bias);
        return PyErr_NoMemory();
    }
    const int64_t *a = (const int64_t *)decay.buf;
    const int64_t *b = (const int64_t *)bias.buf;
    int64_t state = 0;
    for (Py_ssize_t t = 0; t < n; ++t) {
        state = ((a[t] * state + (1 << 15)) >> 16) + b[t];
        out[t] = state;
    }
    PyObject *bytes = PyBytes_FromStringAndSize((const char *)out, n * (Py_ssize_t)sizeof(int64_t));
    free(out);
    PyBuffer_Release(&decay);
    PyBuffer_Release(&bias);
    return bytes;
}

static PyMethodDef methods[] = {
    {"symbol_lut", accel_symbol_lut, METH_VARARGS, "65536-entry uint8 symbol table from an int32 cumulative."},
    {"interleaved_encode", accel_interleaved_encode, METH_VARARGS, "32-lane rANS of (start, freq) pairs in decode order."},
    {"sequential_scan", accel_sequential_scan, METH_VARARGS, "Q15.16 sequential I-SSM scan matching issm.sequential."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_accel", "Native vpxc kernels. Python remains the reference.", -1, methods,
};

PyMODINIT_FUNC PyInit__accel(void) { return PyModule_Create(&moduledef); }
