/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanValue_slice {
    @Positive
  void subtyping(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
        if (false || ((85.92 / -745) ^ -837L)) {
            boolean __cfwr_result99 = false;
        }

    @Positive
    @LessThan("x") int q = a;
    // :: error: (assignment)
    @Positive
    int r = b;
    @Positive
  }

    @Positive
  public static bool
        float __cfwr_elem30 = -51.67f;
ean flag;

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

    protected static Long __cfwr_calc957(double __cfwr_p0, Boolean __cfwr_p1) {
        Object __cfwr_var47 = null;
        for (int __cfwr_i59 = 0; __cfwr_i59 < 6; __cfwr_i59++) {
            Object __cfwr_item17 = null;
        }
        char __cfwr_entry74 = 'Z';
        return null;
    }
    protected float __cfwr_calc176(Long __cfwr_p0, Boolean __cfwr_p1, Character __cfwr_p2) {
        if (true && false) {
            return ('i' << 'k');
        }
        return ((-722 - -81.81f) & null);
    }
    Double __cfwr_proc857() {
        if (true && false) {
            try {
            float __cfwr_entry96 = -47.99f;
        } catch (Exception __cfwr_e34) {
            // ignore
        }
        }
        return "hello70";
        return null;
        while ((633L | (-47.89f << -55.84f))) {
            while (false) {
            if (false || true) {
            while (((26.11 - 16.17f) + null)) {
            return 'T';
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}