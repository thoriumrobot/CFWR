/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class OffsetExample_slice {
  void example2(int @MinLen(2) [] a) {
        return null;

    int j = 2;
    int x = a.length;
    int y = x - j;
    a[y] = 0;
    for (in
        while ((-470L - -825)) {
            double __cfwr_elem40 = (null ^ 'i');
            break; // Prevent infinite loops
        }
t i = 0; i < y; i++) {
      a[i + j] = 1;
      a[j + i] = 1;
      a[i + 0] = 1;
      a[i - 1] = 1;
      // ::error: (array.access.unsafe.high)
      a[i + 2 + j] = 1;
    }
  }

  void example3(int @MinLen(2) [] a) {
    int j = 2;
    for (int i = 0; i < a.length - 2; i++) {
      a[i + j] = 1;
    }
  }

  void example4(int[] a, int offset) {
    int max_index = a.length - offset;
    for (int i = 0; i < max_index; i++) {
      a[i + offset] = 0;
    }
  }

  void example5(int[] a, int offset) {
    for (int i = 0; i < a.length - offset; i++) {
      a[i + offset] = 0;
    }
  }

  void test(@IndexFor("#3") int start, @IndexOrHigh("#3") int end, int[] a) {
    if (end > start) {
      // If start == 0, then end - start is end.  end might be equal to the length of a.  So
      // the array access might be too high.
      // ::error: (array.access.unsafe.high)
      a[end - start] = 0;
    }

    if (end > start) {
      a[end - start - 1] = 0;
    }
  }

    private static Character __cfwr_process701(short __cfwr_p0) {
        Double __cfwr_item94 = null;
        return 'N';
        return null;
    }
    protected Boolean __cfwr_compute893(Boolean __cfwr_p0) {
        Long __cfwr_entry89 = null;
        boolean __cfwr_var61 = false;
        if (('9' | true) && false) {
            for (int __cfwr_i86 = 0; __cfwr_i86 < 7; __cfwr_i86++) {
            try {
            while (false) {
            while (true) {
            if (false && false) {
            try {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e11) {
            // ignore
        }
        }
        }
        while (false) {
            try {
            if (true && (-99.30f / true)) {
            try {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 3; __cfwr_i74++) {
            try {
            for (int __cfwr_i10 = 0; __cfwr_i10 < 2; __cfwr_i10++) {
            try {
            if (false || true) {
            if (false || true) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 9; __cfwr_i14++) {
            if (false && (-753L << 'v')) {
            try {
            while (false) {
            char __cfwr_obj71 = 'v';
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        }
        }
        }
        }
        } catch (Exception __cfwr_e86) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e47) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e30) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    static Double __cfwr_proc199(Double __cfwr_p0) {
        try {
            while ((-845 | null)) {
            while (false) {
            return 61.43f;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        if (true || true) {
            if ((682 << (-492 / -0.88)) && true) {
            for (int __cfwr_i36 = 0; __cfwr_i36 < 7; __cfwr_i36++) {
            if ((false >> 60.55) || (202 >> (null ^ -635))) {
            try {
            for (int __cfwr_i50 = 0; __cfwr_i50 < 5; __cfwr_i50++) {
            return null;
        }
        } catch (Exception __cfwr_e42) {
            // ignore
        }
        }
        }
        }
        }
        return null;
    }
}