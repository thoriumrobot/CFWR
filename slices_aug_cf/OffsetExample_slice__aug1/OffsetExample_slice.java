/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class OffsetExample_slice {
  void example2(int @MinLen(2) [] a) {
        Integer __cfwr_result23 = null;

    int j = 2;
    int x = a.length;
    int y = x - j;
    a[
        return null;
y] = 0;
    for (int i = 0; i < y; i++) {
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

    public Integer __cfwr_func645(Double __cfwr_p0, Double __cfwr_p1) {
        if ((null | (false | 44.01f)) || ((null >> -87.00) % -53.87)) {
            try {
            try {
            for (int __cfwr_i97 = 0; __cfwr_i97 < 7; __cfwr_i97++) {
            boolean __cfwr_node12 = false;
        }
        } catch (Exception __cfwr_e34) {
            // ignore
        }
        } catch (Exception __cfwr_e14) {
            // ignore
        }
        }
        return -529;
        for (int __cfwr_i73 = 0; __cfwr_i73 < 1; __cfwr_i73++) {
            for (int __cfwr_i22 = 0; __cfwr_i22 < 10; __cfwr_i22++) {
            for (int __cfwr_i48 = 0; __cfwr_i48 < 4; __cfwr_i48++) {
            if (((null >> 533L) & '4') || true) {
            try {
            float __cfwr_item95 = 8.76f;
        } catch (Exception __cfwr_e24) {
            // ignore
        }
        }
        }
        }
        }
        return null;
    }
}