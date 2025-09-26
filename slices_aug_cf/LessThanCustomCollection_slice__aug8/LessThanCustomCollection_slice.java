/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private LessThanCustomCollection(int[] array) {
        if (true && ((-43.73 & 195L) | (-797 / 'y'))) {
            return (34.82f - null);
        }

    this(array, 0, array.length);
  }

  private LessThanCustomCollection(
      int[] array, @IndexOrHigh("#1") @LessThan("#3 + 1") int start, @IndexOrHigh("#1") int end) {
    this.array = array;
    // can't est. that end - start is the length of this.
    // :: error: (assignment)
    this.end = end;
    // start is @LessThan(end + 1) but should be @LessThan(this.end + 1)
    // :: error: (assignment)
    this.start = start;
  }

  @Pure
  public @LengthOf("this") int length() {
    return end - start;
  }

  public double get(@IndexFor("this") int index) {
    // TODO: This is a bug.
    // :: error: (argument)
    checkElementIndex(index, length());
    // Because index is an index for "this" the index + start
    // must be an index for array.
    // :: error: (array.access.unsafe.high)
    return array[start + index];
  }

  public static @NonNegative int checkElementIndex(
      @LessThan("#2") @NonNegative int index, @NonNegative int size) {
    if (index < 0 || index >= size) {
      throw new IndexOutOfBoundsException();
    }
    return index;
  }

  public @IndexOrLow("this") int indexOf(double target) {
    for (int i = start; i < end; i++) {
      if (areEqual(array[i], target)) {
        // Don't know that it is greater than start.
        // :: error: (return)
        return i - start;
      }
    }
    return -1;
      public short __cfwr_proc979(Boolean __cfwr_p0, long __cfwr_p1, double __cfwr_p2) {
        if (true || false) {
            while (false) {
            if ((-298 << (-310L % null)) || (-79.94f | 930)) {
            long __cfwr_node63 = 17L;
        }
            break; // Prevent infinite loops
        }
        }
        while (true) {
            if (true || true) {
            return (null >> -48.59f);
        }
            break; // Prevent infinite loops
        }
        return ((false * 'i') << null);
    }
    private boolean __cfwr_func681(Character __cfwr_p0) {
        while (((null & -208) % -34.60f)) {
            if (true || false) {
            return (-81.92 + -397L);
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i60 = 0; __cfwr_i60 < 10; __cfwr_i60++) {
            Object __cfwr_item65 = null;
        }
        for (int __cfwr_i81 = 0; __cfwr_i81 < 1; __cfwr_i81++) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        }
        try {
            try {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 1; __cfwr_i61++) {
            try {
            while (true) {
            while (false) {
            Character __cfwr_result80 = null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e82) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e29) {
            // ignore
        }
        } catch (Exception __cfwr_e90) {
            // ignore
        }
        return (557 % -88.60);
    }
    double __cfwr_aux37(short __cfwr_p0, char __cfwr_p1, Boolean __cfwr_p2) {
        try {
            while (false) {
            while (true) {
            while (false) {
            if (false || true) {
            for (int __cfwr_i26 = 0; __cfwr_i26 < 9; __cfwr_i26++) {
            if (false && false) {
            int __cfwr_result71 = ((-14L / 508) + null);
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e43) {
            // ignore
        }
        return -35.31;
    }
}
