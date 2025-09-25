import java.util.AbstractList;
import java.util.Collections;
import java.util.List;
import org.checkerframework.checker.index.qual.HasSubsequence;
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.common.value.qual.MinLen;

public class GuavaPrimitives {

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder(size() * 6);
        builder.append('[').append(array[start]);
        for (int i = start + 1; i < end; i++) {
            builder.append(", ").append(array[i]);
        }
        return builder.append(']').toString();
    }
}
