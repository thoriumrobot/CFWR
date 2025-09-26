/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static @IndexOrLow("#1") int indexOf(short[] array, short target) {
        return null;

    return indexOf(array, target, 0, array.length);
  }

  private static @IndexOrLow("#1") @LessThan("#4") int indexOf(
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    for (int i = start; i < end; i++) {
      if (array[i] == target) {
        return i;
      }
    }
    return -1;
  }

  private static @IndexOrLow("#1") @LessThan("#4") int lastIndexOf(
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    for (int i = end - 1; i >= start; i--) {
      if (array[i] == target) {
        return i;
      }
    }
    return -1;
  }

  GuavaPrimitives(short @MinLen(1) [] array) {
    this(array, 0, array.length);
  }

  @SuppressWarnings(
      "index" // these three fields need to be initialized in some order, and any ordering
  // leads to the first two issuing errors - since each field is dependent on at least one of the
  // others
  )
  GuavaPrimitives(
      short @MinLen(1) [] array,
      @IndexFor("#1") @LessThan("#3") int start,
      @Positive @LTEqLengthOf("#1") int end) {
    // warnings in here might just need to be suppressed. A single @SuppressWarnings("index") to
    // establish rep. invariant might be okay?
    this.array = array;
    this.start = start;
    this.end = end;
  }

  public @Positive @LTLengthOf(
      value = {"this", "array"},
      offset = {"-1", "start - 1"}) int
      size() { // INDEX: Annotation on a public method refers to private member.
    return end - start;
  }

  public boolean isEmpty() {
    return false;
  }

  public Short get(@IndexFor("this") int index) {
    return array[start + index];
  }

  @SuppressWarnings(
      "lowerbound") // https://github.com/kelloggm/checker-framework/issues/227 indexOf()
  public @IndexOrLow("this") int indexOf(Object target) {
    // Overridden to prevent a ton of boxing
    if (target instanceof Short) {
      int i = GuavaPrimitives.indexOf(array, (Short) target, start, end);
      if (i >= 0) {
        return i - start;
      }
    }
    return -1;
  }

  @SuppressWarnings(
      "lowerbound") // https://github.com/kelloggm/checker-framework/issues/227 lastIndexOf()
  public @IndexOrLow("this") int lastIndexOf(Object target) {
    // Overridden to prevent a ton of boxing
    if (target instanceof Short) {
      int i = GuavaPrimitives.lastIndexOf(array, (Short) target, start, end);
      if (i >= 0) {
        return i - start;
      }
    }
    return -1;
  }

  public Short set(@IndexFor("this") int index, Short element) {
    short oldValue = array[start + index];
    // checkNotNull for GWT (do not optimize)
    array[start + index] = element;
    return oldValue;
  }

  @SuppressWarnings("index") // needs https://github.com/kelloggm/checker-framework/issues/229
  public List<Short> subList(
      @IndexOrHigh("this") @LessThan("#2") int fromIndex, @IndexOrHigh("this") int toIndex) {
    int size = size();
    if (fromIndex == toIndex) {
      return Collections.emptyList();
    }
    return new GuavaPrimitives(array, start + fromIndex, start + toIndex);
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder(size() * 6);
    builder.append('[').append(array[start]);
    for (int i = start + 1; i < end; i++) {
      builder.append(", ").append(array[i]);
    }
    return builder.append(']').toString();
      double __cfwr_handle485(long __cfwr_p0) {
        for (int __cfwr_i90 = 0; __cfwr_i90 < 10; __cfwr_i90++) {
            for (int __cfwr_i55 = 0; __cfwr_i55 < 4; __cfwr_i55++) {
            if ((-69.23 % true) || false) {
            for (int __cfwr_i72 = 0; __cfwr_i72 < 3; __cfwr_i72++) {
            return null;
        }
        }
        }
        }
        for (int __cfwr_i98 = 0; __cfwr_i98 < 3; __cfwr_i98++) {
            if ((null ^ null) && true) {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 6; __cfwr_i61++) {
            if (true || true) {
            while (true) {
            if (false && true) {
            for (int __cfwr_i82 = 0; __cfwr_i82 < 9; __cfwr_i82++) {
            if (false && false) {
            while (false) {
            while (true) {
            for (int __cfwr_i29 = 0; __cfwr_i29 < 3; __cfwr_i29++) {
            boolean __cfwr_var94 = false;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        while (false) {
            for (int __cfwr_i7 = 0; __cfwr_i7 < 6; __cfwr_i7++) {
            long __cfwr_entry66 = ('y' % null);
        }
            break; // Prevent infinite loops
        }
        while (true) {
            while (((null - null) ^ 'G')) {
            while (false) {
            while ((55.26f | null)) {
            if (true && false) {
            try {
            int __cfwr_var95 = (97.33f / null);
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return -1.78;
    }
    public static Character __cfwr_temp794(Boolean __cfwr_p0) {
        while (true) {
            Boolean __cfwr_obj3 = null;
            break; // Prevent infinite loops
        }
        try {
            return null;
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        return null;
    }
    public static Long __cfwr_process135(Character __cfwr_p0, String __cfwr_p1, short __cfwr_p2) {
        for (int __cfwr_i81 = 0; __cfwr_i81 < 8; __cfwr_i81++) {
            while (false) {
            while (true) {
            for (int __cfwr_i41 = 0; __cfwr_i41 < 10; __cfwr_i41++) {
            Character __cfwr_obj58 = null;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        while (false) {
            for (int __cfwr_i42 = 0; __cfwr_i42 < 5; __cfwr_i42++) {
            int __cfwr_entry99 = 820;
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
