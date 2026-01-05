/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class OffsetExample_slice {
    @Positive
  void example2(int @MinLen(2) [] a) {
        for (int __cfwr_i41 = 0; __cfwr_i41 < 3; __cfwr_i41++) {
            int __cfwr_item5 = -581;
        }

    @Positive
    int j = 2;
    @Positive
    int x = a.length;
    @Positive
    int y = x - j;
    @Positive
    a[y] = 0;
    @Positive
    for (int i = 0; i < y; i++) {
    @Positive
      a[i + j] = 1;
    @Positive
      a[j + i] = 1;
    @Positive
      a[i + 0] = 1;
    @Positive
      a[i - 1] = 1;
      // ::error: (array.access.unsafe.high)
    @Positive
      a[i + 2 + j] = 1;
    @Positive
    }
    @Positive
  }

    @Positive
  void example3(int @MinLen(2) [] a) {
    @Positive
    int j = 2;
    @Positive
    for (int i = 0; i < a.length - 2; i++) {
    @Positive
      a[i + j] = 1;
    @Positive
    }
    @Positive
  }

    Object __cfwr_handle941() {
        if (false || true) {
            return null;
        }
        if (true && (-246L >> (18.47 & null))) {
            return null;
        }
        while ((-78.12f * 52.77f)) {
            while (true) {
            Boolean __cfwr_elem97 = null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i40 = 0; __cfwr_i40 < 7; __cfwr_i40++) {
            short __cfwr_result72 = null;
        }
        return null;
    }
    private Integer __cfwr_helper467(float __cfwr_p0) {
        if ((-24.54f & (true ^ -786L)) || (null % null)) {
            if (true || true) {
            return false;
        }
        }
        char __cfwr_node38 = '8';
        return null;
    }
    static Long __cfwr_func944(int __cfwr_p0, double __cfwr_p1) {
        double __cfwr_result87 = 17.94;
        if (false || (165L + null)) {
            Integer __cfwr_entry38 = null;
        }
        return null;
    }
}