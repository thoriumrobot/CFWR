/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class OffsetExample_slice {
    @Positive
  void example2(int @MinLen(2) [] a) {
        while (true) {
            for (int __cfwr_i98 = 0; __cfwr_i98 < 7; __cfwr_i98++) {
            if (false && ('z' + -14L)) {
            while (false) {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 6; __cfwr_i32++) {
            if (true && false) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
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

    protected byte __cfwr_aux749(byte __cfwr_p0, Object __cfwr_p1, Character __cfwr_p2) {
        return null;
        return null;
        try {
            for (int __cfwr_i66 = 0; __cfwr_i66 < 3; __cfwr_i66++) {
            while (false) {
            for (int __cfwr_i39 = 0; __cfwr_i39 < 7; __cfwr_i39++) {
            try {
            try {
            Long __cfwr_node73 = null;
        } catch (Exception __cfwr_e82) {
            // ignore
        }
        } catch (Exception __cfwr_e4) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e82) {
            // ignore
        }
        if (false && false) {
            return true;
        }
        return null;
    }
    float __cfwr_handle22(Integer __cfwr_p0, int __cfwr_p1, Integer __cfwr_p2) {
        float __cfwr_temp99 = 60.65f;
        Character __cfwr_result77 = null;
        Double __cfwr_entry49 = null;
        Long __cfwr_node68 = null;
        return (null - 64.73);
    }
}