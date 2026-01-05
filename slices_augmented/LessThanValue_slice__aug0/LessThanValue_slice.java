/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanValue_slice {
    @Positive
  void subtyping(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
        long __cfwr_obj14 = -971L;

    @Positive
    @LessThan("x") int q = a;
    // :: error: (assignment)
    @Positive
    int r = b;
    @Positive
  }

    @Positive
  public static boolean flag;

    @Positive
  void lub(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
    @Positive
    @LessThan("x") int r = flag ? a : b;
    // :: error: (assignment)
    @Positive
    int s = flag ? a : b;
    @Positive
  }

    @Positive
  void transitive(int a, int b, int c) {
    @Positive
    if (a < b) {
    @Positive
      if (b < c) {
        // :: error: (assignment)
    @Positive
        @LessThan("c") int x = a;
    @Positive
      }
    @Positive
    }
    @Positive
  }

    @Positive
  void calls() {
    @Positive
    isLessThan(0, 1);
    @Positive
    isLessThanOrEqual(0, 0);
    @Positive
  }

    @Positive
  void isLessThan(@LessThan("#2") @NonNegative int start, int end) {
    @Positive
    @NonNegative int x = end - start - 1;
    @Positive
    @Positive int y = end - start;
    @Positive
  }

    @Positive
  @NonNegative int isLessThanOrEqual(@LessThan("#2 + 1") @NonNegative int start, int end) {
    @Positive
    return end - start;
    @Positive
  }

    @Positive
  public void setMaximumItemCount(int maximum) {
    @Positive
    if (maximum < 0) {
    @Positive
      throw new IllegalArgumentException("Negative 'maximum' argument.");
    @Positive
    }
    @Positive
    int count = getCount();
    @Positive
    if (count > maximum) {
    @Positive
      @Positive int y = count - maximum;
    @Positive
      @NonNegative int deleteIndex = count - maximum - 1;
    @Positive
    }
    @Positive
  }

    @Positive
  int getCount() {
    @Positive
    throw new RuntimeException();
    @Positive
  }

    @Positive
  void method(@NonNegative int m) {
    @Positive
    boolean[] has_modulus = new boolean[m];
    @Positive
    @LessThan("m") int x = foo(m);
    @Positive
    @IndexFor("has_modulus") int rem = foo(m);
    @Positive
  }

    @Positive
  @LessThan("#1") @NonNegative int foo(int in) {
    @Positive
    throw new RuntimeException();
    @Positive
  }

    private String __cfwr_func330(Float __cfwr_p0, double __cfwr_p1) {
        try {
            for (int __cfwr_i57 = 0; __cfwr_i57 < 7; __cfwr_i57++) {
            try {
            while (true) {
            for (int __cfwr_i66 = 0; __cfwr_i66 < 7; __cfwr_i66++) {
            while (true) {
            return (null >> null);
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e30) {
            // ignore
        }
        if (false && false) {
            for (int __cfwr_i73 = 0; __cfwr_i73 < 1; __cfwr_i73++) {
            try {
            for (int __cfwr_i65 = 0; __cfwr_i65 < 8; __cfwr_i65++) {
            try {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 5; __cfwr_i12++) {
            try {
            int __cfwr_node28 = ((995L | 131L) % (null / -880L));
        } catch (Exception __cfwr_e3) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e61) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e81) {
            // ignore
        }
        }
        }
        return 47.63;
        return "temp66";
    }
    private static float __cfwr_process901(String __cfwr_p0) {
        return 90L;
        Float __cfwr_entry48 = null;
        return (-183L >> -544L);
    }
}